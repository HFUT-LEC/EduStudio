from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class LPKT_S(GDBaseModel):
    default_cfg = {
        'd_a': 50,
        'd_k': 128,
        'd_e': 128,
        'd_s': 128, 
        'q_gamma': 0.03,
        'drop_rate': 0.2,
        'param_init_type': 'xavier_uniform'
    }

    def build_cfg(self):
        self.n_stu = self.datatpl_cfg['dt_info']['stu_count']
        self.n_exer = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.n_at = self.datatpl_cfg['dt_info']['answer_time_count'] # answer time
        self.n_it = self.datatpl_cfg['dt_info']['interval_time_count'] # interval time
        self.d_k = self.modeltpl_cfg['d_k']
        self.d_e = self.modeltpl_cfg['d_e']
        self.d_a = self.modeltpl_cfg['d_a']
        self.d_s = self.modeltpl_cfg['d_s']

    def add_extra_data(self, **kwargs):
        Q_mat = kwargs['Q_mat']
        Q_mat[Q_mat == 0] = self.modeltpl_cfg['q_gamma']
        self.q_matrix = Q_mat.to(self.device).float()

    def build_model(self):
        d_k = self.modeltpl_cfg['d_k']
        d_e = self.modeltpl_cfg['d_e']
        d_a = self.modeltpl_cfg['d_a']
        d_s = self.modeltpl_cfg['d_s']

        self.at_embed = nn.Embedding(self.n_at, d_k)
        self.it_embed = nn.Embedding(self.n_it, d_k)
        self.exer_embed = nn.Embedding(self.n_exer, d_e)
        self.stu_embed = nn.Embedding(self.n_stu, d_s)

        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        self.linear_4 = nn.Linear(4 * d_k, d_k)
        self.linear_5 = nn.Linear(3 * d_k, d_k)

        self.dropout = nn.Dropout(self.modeltpl_cfg['drop_rate'])

    def forward(self, stu_id, exer_seq, answer_time_seq, interval_time_seq, mask_seq, **kwargs):  # answer_time_seq是学生回答每道习题的时间
        #  interval_time_seq是学生回答习题间的间隔时间
        a_data = exer_seq
        batch_size, seq_len = exer_seq.shape
        e_embed_data = self.exer_embed(exer_seq)
        at_embed_data = self.at_embed(answer_time_seq)
        it_embed_data = self.it_embed(interval_time_seq)
        stu_seq = stu_id.unsqueeze(1).repeat(1,seq_len) * mask_seq
        stu_embed_data = self.stu_embed(stu_seq)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)  
        # view相当于reshape,view(-1,1)即让矩阵第二维为1，第一维自适应地改变
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_cpt, self.d_k)).repeat(batch_size, 1, 1).to(self.device)
        h_tilde_pre = None
        all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))  
        # 2是指在第三个维度上进行拼接，这里可能得到论文中的lt-1或lt
        learning_pre = torch.zeros(batch_size, self.d_k).to(self.device)

        pred = torch.zeros(batch_size, seq_len).to(self.device)
        for t in range(0, seq_len - 1):
            e = exer_seq[:, t]  # shape:batch_size
            # q_e: (bs, 1, n_skill)
            q_e = self.q_matrix[e].view(batch_size, 1, -1)  # 习题e对应的知识点
            it = it_embed_data[:, t]
            stu = stu_embed_data[:, t]

            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            learning = all_learning[:, t]
            learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))  # 对应论文公式4
            learning_gain = learning_gain.tanh()  # 对应论文公式4
            gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, stu), 1))  # 对应论文公式12
            gamma_l = gamma_l.sigmoid()  # 对应论文公式12
            LG = gamma_l * ((learning_gain + 1) / 2)  # 对应论文公式13
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))  # 对应论文公式13

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            gamma_f = self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                stu.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)).sigmoid()  # 对应论文公式14
            pro = LG_tilde - gamma_f * h_pre  # 对应论文公式15
            h = pro + h_pre  # 对应论文公式15

            # Predicting Module
            h_tilde = self.q_matrix[exer_seq[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)  # 
            y = self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde, stu), 1)).sigmoid().sum(1) / self.d_k  # 对应论文公式16
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde
        return pred 

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)[:, 0:-1]
        # y_pd = y_pd[:, :-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def get_main_loss(self, **kwargs):
        y_pd = self(**kwargs)[:,0:-1] # 可能与原文不符合，原文是[:,1:]
        # y_pd = y_pd[:, :-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
