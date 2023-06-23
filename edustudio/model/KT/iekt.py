r"""
IEKT
##########################################

Reference:
    Ting Long et al. "Tracing Knowledge State with Individual Cognition and Acquisition Estimation" in SIGIR 2021.

Reference Code:
    https://github.com/ApexEDM/iekt

"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..utils.components import MLP
from torch.distributions import Categorical


class mygru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, input_dim, hidden_dim):
        super().__init__()
        
        this_layer = n_layer
        self.g_ir = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_iz = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_in = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_hr = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.g_hz = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.g_hn = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, h):
        r_t = self.sigmoid(
            self.g_ir(x) + self.g_hr(h)
        )
        z_t = self.sigmoid(
            self.g_iz(x) + self.g_hz(h)
        )
        n_t = self.tanh(
            self.g_in(x) + self.g_hn(h).mul(r_t)
        )
        h_t = (1 - z_t) * n_t + z_t * h
        return h_t
    
class funcsgru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class IEKT(GDBaseModel):
    default_cfg = {
        'd_q': 64, # embedding size of exercise embedding
        'd_c': 64, # embedding size of concept embedding
        'd_m': 128, # embedding size of cognition levels matrix
        'd_s': 128, # embedding size of acquisition sensitivity levels matrix
        'd_h': 64, # size of hidden state
        'n_cog_level': 10, 
        'n_acq_level': 10,
        'dropout_rate': 0.0,
        'gamma': 0.93,
        'lambda': 40.0,
    }

    def build_cfg(self):
        self.n_stu = self.datatpl_cfg['dt_info']['stu_count']
        self.n_exer = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.window_size = self.datatpl_cfg['dt_info']['real_window_size']
        self.n_cog_level = self.modeltpl_cfg['n_cog_level']
        self.n_acq_level = self.modeltpl_cfg['n_acq_level']
        self.d_v = self.modeltpl_cfg['d_h'] + self.modeltpl_cfg['d_q'] + self.modeltpl_cfg['d_c']
        self.d_r = self.d_v + self.modeltpl_cfg['d_m']
        self.d_i = self.modeltpl_cfg['d_q'] + self.modeltpl_cfg['d_c'] + self.modeltpl_cfg['d_s']

    def build_model(self):
        self.exer_emb = nn.Embedding(self.n_exer, self.modeltpl_cfg['d_q'])
        self.cpt_emb = nn.Embedding(self.n_cpt, self.modeltpl_cfg['d_c'])
        self.cog_matrix = nn.Embedding(self.n_cog_level, self.modeltpl_cfg['d_m'])
        self.acq_matrix = nn.Embedding(self.n_acq_level, self.modeltpl_cfg['d_s'])

        self.pd_layer = MLP(
            input_dim=self.d_r, output_dim=1, 
            dnn_units=[self.d_r], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )
        self.f_p = MLP(
            input_dim=self.d_v, output_dim=self.n_cog_level, 
            dnn_units=[self.d_r], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )
        self.f_e = MLP(
            input_dim=self.d_v * 4, output_dim=self.n_acq_level,
            dnn_units=[self.d_r], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )
        self.gru_h = mygru(0, self.d_i, self.modeltpl_cfg['d_h'])

    def get_exer_representation(self, exer_ids, cpt_seq, cpt_seq_mask):
        exer_emb = self.exer_emb(exer_ids) # windows_size x emb size
        cpt_emb = torch.vstack(
            [self.cpt_emb(cpt_seq[i])[cpt_seq_mask[i] == 1].mean(dim=0) for i in range(cpt_seq.shape[0])]
        )
        return torch.cat([exer_emb, cpt_emb], dim = -1)

    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.f_p(x), dim = softmax_dim)
    
    def pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.f_e(x), dim = softmax_dim)
    
    def update_state(self, h, v, s_t, operate):
        v_cat = torch.cat([
            v.mul(operate.reshape(-1,1)),
            v.mul((1 - operate).reshape(-1,1))], dim = 1)
        e_cat = torch.cat([
            s_t.mul((1-operate).reshape(-1,1)),
            s_t.mul((operate).reshape(-1,1))], dim = 1)
        inputs = v_cat + e_cat
        next_p_state = self.gru_h(inputs, h)
        return next_p_state
     
    def get_main_loss(self, **kwargs):
        exer_seq = kwargs['exer_seq']
        exer_mask_seq = kwargs['mask_seq']
        cpt_seq = kwargs['cpt_seq']
        cpt_seq_mask = kwargs['cpt_seq_mask']
        label_seq = kwargs['label_seq']

        batch_size = exer_seq.shape[0]
        seq_len = exer_seq.shape[1]
        h = torch.zeros(batch_size, self.modeltpl_cfg['d_h']).to(self.device)
        p_action_list, pre_state_list, emb_action_list, states_list, reward_list, predict_list, ground_truth_list = [], [], [], [], [], [], []


        for t in range(seq_len):
            # read stage
            exer_seq_col = exer_seq[:, t]
            cpt_seq_col = cpt_seq[:, t]
            cpt_seq_mask_col = cpt_seq_mask[:, t]
            label_seq_col = label_seq[:, t]
            v = self.get_exer_representation(exer_seq_col,  cpt_seq_col, cpt_seq_mask_col)
            h_v = torch.cat([h, v], dim=1)
            v_h = torch.cat([v, h], dim=1)
            flip_prob_emb = self.pi_cog_func(v_h)

            m = Categorical(flip_prob_emb) # 从多项式分布中sample
            cog_sample = m.sample()
            m_t = self.cog_matrix(cog_sample) # 从CE矩阵取出对应emb
            
            logits = self.pd_layer(torch.cat([h_v, m_t], dim = 1)) # 得到预测结果
            prob = logits.sigmoid()

            # write stage
            v_g = torch.cat([
                h_v.mul(label_seq_col.reshape(-1,1).float()),
                h_v.mul((1-label_seq_col).reshape(-1,1).float())],
                dim = 1)

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))  # 将预测值离散化
            v_p = torch.cat([
                h_v.mul(out_operate_logits.reshape(-1,1).float()),
                h_v.mul((1-out_operate_logits).reshape(-1,1).float())],
                dim = 1)                
            v_m = torch.cat([v_g, v_p], dim = 1)

            flip_prob_emb = self.pi_sens_func(v_m)
            m = Categorical(flip_prob_emb)
            acq_sample = m.sample()
            s_t = self.acq_matrix(acq_sample)
            h = self.update_state(h, v, s_t, label_seq_col)

            emb_action_list.append(cog_sample)
            p_action_list.append(acq_sample)
            states_list.append(v_m)
            pre_state_list.append(v_h)
            
            ground_truth_list.append(label_seq_col)
            predict_list.append(logits.squeeze(1))
            this_reward = torch.where(out_operate_logits.squeeze(1).float() == label_seq_col,
                            torch.tensor(1).to(self.device), 
                            torch.tensor(0).to(self.device))
            reward_list.append(this_reward)

        # RL 训练
        seq_num = exer_mask_seq.sum(dim=1)
        emb_action_tensor = torch.stack(emb_action_list, dim = 1) # ac的sample操作
        p_action_tensor = torch.stack(p_action_list, dim = 1) # ce的sample操作
        state_tensor = torch.stack(states_list, dim = 1) # 输出的groudtruth和predict合并
        pre_state_tensor = torch.stack(pre_state_list, dim = 1)
        reward_tensor = torch.stack(reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, self.window_size)).float()
        logits_tensor = torch.stack(predict_list, dim = 1)
        ground_truth_tensor = torch.stack(ground_truth_list, dim = 1)
        loss = []
        tracat_logits = []
        tracat_ground_truth = []
        
        for i in range(0, batch_size):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
        
            this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)

            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.modeltpl_cfg['gamma'] * advantage + delta_t[0]
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)
            
            pi_cog = self.pi_cog_func(this_cog_state[:-1])
            pi_a_cog = pi_cog.gather(1,p_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog
            
            loss.append(torch.sum(loss_cog))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                advantage = self.modeltpl_cfg['gamma'] * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)
            
            pi_sens = self.pi_sens_func(this_sens_state[:-1])
            pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_sens = - torch.log(pi_a_sens) * advantage_sens
            loss.append(torch.sum(loss_sens))
            
            this_prob = logits_tensor[i][0: this_seq_len]
            this_groud_truth = ground_truth_tensor[i][0: this_seq_len]

            tracat_logits.append(this_prob)
            tracat_ground_truth.append(this_groud_truth)
        

        bce = F.binary_cross_entropy_with_logits(torch.cat(tracat_logits, dim = 0), torch.cat(tracat_ground_truth, dim = 0))
           
        label_len = torch.cat(tracat_ground_truth, dim = 0).size()[0]
        loss_l = sum(loss)
        loss = self.modeltpl_cfg['lambda'] * (loss_l / label_len) +  bce
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)


    @torch.no_grad()
    def predict(self, **kwargs):
        exer_seq = kwargs['exer_seq']
        exer_mask_seq = kwargs['mask_seq']
        cpt_seq = kwargs['cpt_seq']
        cpt_seq_mask = kwargs['cpt_seq_mask']
        label_seq = kwargs['label_seq']

        batch_size = exer_seq.shape[0]
        seq_len = exer_seq.shape[1]
        h = torch.zeros(batch_size, self.modeltpl_cfg['d_h']).to(self.device)
        batch_probs, uni_prob_list =[], []


        for t in range(seq_len):
            # read stage
            exer_seq_col = exer_seq[:, t]
            cpt_seq_col = cpt_seq[:, t]
            cpt_seq_mask_col = cpt_seq_mask[:, t]
            label_seq_col = label_seq[:, t]
            v = self.get_exer_representation(exer_seq_col,  cpt_seq_col, cpt_seq_mask_col)
            h_v = torch.cat([h, v], dim=1)
            v_h = torch.cat([v, h], dim=1)
            flip_prob_emb = self.pi_cog_func(v_h)

            m = Categorical(flip_prob_emb) # 从多项式分布中sample
            cog_sample = m.sample()
            m_t = self.cog_matrix(cog_sample) # 从CE矩阵取出对应emb
            
            logits = self.pd_layer(torch.cat([h_v, m_t], dim = 1)) # 得到预测结果
            prob = logits.sigmoid()

            # write stage
            v_g = torch.cat([
                h_v.mul(label_seq_col.reshape(-1,1).float()),
                h_v.mul((1-label_seq_col).reshape(-1,1).float())],
                dim = 1)

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))  # 将预测值离散化
            v_p = torch.cat([
                h_v.mul(out_operate_logits.reshape(-1,1).float()),
                h_v.mul((1-out_operate_logits).reshape(-1,1).float())],
                dim = 1)                
            v_m = torch.cat([v_g, v_p], dim = 1)

            flip_prob_emb = self.pi_sens_func(v_m)
            m = Categorical(flip_prob_emb)
            acq_sample = m.sample()
            s_t = self.acq_matrix(acq_sample)
            h = self.update_state(h, v, s_t, label_seq_col)

            uni_prob_list.append(prob.detach())

        seq_num = exer_mask_seq.sum(dim=1)
        prob_tensor = torch.cat(uni_prob_list, dim = 1)
        for i in range(0, batch_size):
            this_seq_len = seq_num[i]
            batch_probs.append(prob_tensor[i][0: this_seq_len])
        y_pd = torch.cat(batch_probs, dim = 0)

        y_gt = kwargs['label_seq']
        y_gt = y_gt[kwargs['mask_seq'] == 1]
        
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }
