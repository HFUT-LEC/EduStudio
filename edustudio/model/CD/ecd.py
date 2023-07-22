r"""
ECD
##########################################

Reference:
    Yuqiang Zhou et al. "Modeling Context-aware Features for Cognitive Diagnosis in Student Learning" in KDD 2021.


"""
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
from ..utils.components import PosMLP
import torch
import torch.nn.functional as F

class ECD(GDBaseModel):
    r"""
    ECD

    default_cfg:
        'dnn_units': [512, 256]  # dimension list of hidden layer in prediction layer
        'dropout_rate': 0.5      # dropout rate
        'activation': 'sigmoid'  # activation function in prediction layer
        'disc_scale': 10         # discrimination scale
        'emb_dim': 10           # dimension of q,k
        'con_dim': 1            # dimension of v
    """
    default_cfg = {
        'dnn_units': [512, 256],
        'dropout_rate': 0.5,
        'activation': 'sigmoid',
        'disc_scale': 10,
        'emb_dim': 10,
        'con_dim': 1,
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.qk_dim = self.modeltpl_cfg["emb_dim"]
        self.con_dim = self.modeltpl_cfg["con_dim"]
        self.batch_size = self.traintpl_cfg['batch_size']

    def build_model(self):
        self.con_ans_num = self.datatpl_cfg['dt_info']['qqq_count']
        self.stu_q = nn.Embedding(self.n_user, self.qk_dim, padding_idx=0)
        self.qqq_group = [torch.tensor(k).to(self.device) for k in self.qqq_group_list]
        self.group_len = len(self.qqq_group)
        self.con_ans_v = nn.Embedding(self.con_ans_num, self.con_dim, padding_idx=0)
        self.con_ans_k = nn.Embedding(self.con_ans_num, self.qk_dim, padding_idx=0)

        self.d_t = nn.Embedding(self.n_user, 1, padding_idx=0)

    def add_extra_data(self, **kwargs):
        self.qqq_group_list = kwargs['qqq_group_list']
        self.Q_mat = kwargs['Q_mat'].to(self.device)
        self.QQQ_list = kwargs['qqq_list'].to(self.device)

    def atten(self, query, key, value):
        v_dim = value.shape[2]
        btch = query.shape[0]
        temp0 = torch.matmul(query, key.transpose(1, 2))
        temp_query_n = query.norm(dim=2).view(btch, -1, 1)
        temp_key_n = key.norm(dim=2).view(btch, 1, -1)
        temp_n = temp_query_n * temp_key_n
        temp_sim = temp0 / temp_n
        sim = F.softmax(temp_sim, dim=2).transpose(1, 2)
        res = torch.matmul(value.transpose(1, 2), sim).transpose(1, 2).view(btch, -1, v_dim)
        return res

    def self_atten(self, query, key, value):
        v_dim = value.shape[2]
        btch = query.shape[0]
        temp0 = torch.matmul(query, key.transpose(1, 2))
        temp_query_n = query.norm(dim=2).view(btch, -1, 1)
        temp_key_n = key.norm(dim=2).view(btch, 1, -1)
        temp_n = temp_query_n * temp_key_n
        temp_sim = temp0 / temp_n
        sim = F.softmax(temp_sim, dim=2).transpose(1, 2)
        res = torch.matmul(value.transpose(1, 2), sim).transpose(1, 2).view(btch, -1, v_dim)
        return res

    def forward(self, stu_id, qqq_ids):
        # before prednet
        group_v = torch.tensor([]).type(torch.FloatTensor).to(self.device)
        group_k = torch.tensor([]).type(torch.FloatTensor).to(self.device)
        btch = stu_id.shape[0]
        stu_q = self.stu_q(stu_id).view(btch, 1, -1)
        for i in range(self.group_len):
            temp_query = stu_q
            temp_idx = qqq_ids[:, self.qqq_group[i]]
            temp_v = self.con_ans_v(temp_idx)
            temp_k = self.con_ans_k(temp_idx)
            temp_value = self.atten(temp_query, temp_k, temp_v)
            temp_key = self.atten(temp_query, temp_k, temp_k)
            group_v = torch.cat([group_v, temp_value], dim=1)
            group_k = torch.cat([group_k, temp_key], dim=1)
        group_value = self.self_atten(group_k, group_k, group_v)
        group_key = self.self_atten(group_k, group_k, group_k)
        temp_query = stu_q
        context_emb = self.atten(temp_query, group_key, group_value)
        context_emb = torch.sigmoid(context_emb.view(btch, self.con_dim))
        dt_w = torch.sigmoid(self.d_t(stu_id)).view(btch, 1)
        return context_emb, dt_w

class ECD_IRT(ECD):
    default_cfg = {
        "a_range": -1.0,  # disc range
        "diff_range": -1.0,  # diff range
        "fix_a": False,
        "fix_c": True,
        'loss_weight': [4, 0],
    }
    def build_model(self):
        super().build_model()
        self.theta = nn.Embedding(self.n_user, 1)  # student ability
        self.a = 0.0 if self.modeltpl_cfg['fix_a'] else nn.Embedding(self.n_item, 1)  # exer discrimination
        self.b = nn.Embedding(self.n_item, 1)  # exer difficulty

    @staticmethod
    def irf(theta, a, b,  D=1.702):
        return 1 / (1 + torch.exp(-D * a * (theta - b)))

    def forward(self, stu_id, exer_id):
        items_Q_mat = self.Q_mat[exer_id]
        stus_qqq = self.QQQ_list[stu_id]
        theta_inner = self.theta(stu_id)
        a = self.a(exer_id)
        b = self.b(exer_id)
        theta_context, dt_w = super().forward(stu_id, stus_qqq)
        theta_all = theta_context * dt_w + theta_inner * (1 - dt_w)
        if self.modeltpl_cfg['diff_range'] is not None:
            b = self.modeltpl_cfg['diff_range'] * (torch.sigmoid(b) - 0.5)
        if self.modeltpl_cfg['a_range'] is not None:
            a = self.modeltpl_cfg['a_range'] * torch.sigmoid(a)
        else:
            a = F.softplus(a) # 让区分度大于0，保持单调性假设
        if torch.max(theta_inner != theta_inner) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The diff_range or a_range is too large.')
        pre_all = self.irf(theta_all, a, b)
        pre_context = self.irf(theta_context, a, b)
        pre_inner = self.irf(theta_inner, a, b)
        if self.training:
            return pre_all, pre_context, pre_inner
        else:
            return pre_all

    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd_all, pd_context, pd_inner = self(stu_id, exer_id)
        pd_all, pd_context, pd_inner = pd_all.flatten(), pd_context.flatten(), pd_inner.flatten()
        loss_all = F.binary_cross_entropy(input=pd_all, target=label)
        loss_context = F.binary_cross_entropy(input=pd_context, target=label)
        loss_inner = F.binary_cross_entropy(input=pd_inner, target=label)
        w = self.modeltpl_cfg['loss_weight']
        # loss = loss_all+w[0]*loss_context+w[1]*loss_inner
        return {
            'loss_main': loss_all,
            'loss_context': w[0] * loss_context,
            'loss_inner': w[1] * loss_inner
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)

class ECD_MIRT(ECD):
    default_cfg = {
        "a_range": -1.0,  # disc range
        "emb_dim": 32,
        'loss_weight': [0, 4],
    }
    def build_model(self):
        super().build_model()
        self.emb_dim = self.modeltpl_cfg['emb_dim']
        self.theta = nn.Embedding(self.n_user, self.emb_dim)  # student ability
        self.a = nn.Embedding(self.n_item, self.emb_dim)  # exer discrimination
        self.b = nn.Embedding(self.n_item, 1)  # exer intercept term

    @staticmethod
    def irf(theta, a, b):
        return 1 / (1 + torch.exp(- torch.sum(torch.multiply(a, theta), axis=-1) + b))

    def forward(self, stu_id, exer_id):
        items_Q_mat = self.Q_mat[exer_id]
        stus_qqq = self.QQQ_list[stu_id]
        theta_inner = self.theta(stu_id)
        a = self.a(exer_id)
        b = self.b(exer_id).flatten()
        theta_context, dt_w = super().forward(stu_id, stus_qqq)
        theta_all = theta_context * dt_w + theta_inner * (1 - dt_w)
        if self.modeltpl_cfg['a_range'] is not None:
            a = self.modeltpl_cfg['a_range'] * torch.sigmoid(a)
        else:
            a = F.softplus(a) # 让区分度大于0，保持单调性假设
        if torch.max(theta_inner != theta_inner) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The diff_range or a_range is too large.')
        pre_all = self.irf(theta_all, a, b)
        pre_context = self.irf(theta_context, a, b)
        pre_inner = self.irf(theta_inner, a, b)
        if self.training:
            return pre_all, pre_context, pre_inner
        else:
            return pre_all

    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd_all, pd_context, pd_inner = self(stu_id, exer_id)
        pd_all, pd_context, pd_inner = pd_all.flatten(), pd_context.flatten(), pd_inner.flatten()
        loss_all = F.binary_cross_entropy(input=pd_all, target=label)
        loss_context = F.binary_cross_entropy(input=pd_context, target=label)
        loss_inner = F.binary_cross_entropy(input=pd_inner, target=label)
        w = self.modeltpl_cfg['loss_weight']
        # loss = loss_all+w[0]*loss_context+w[1]*loss_inner
        return {
            'loss_main': loss_all,
            'loss_context': w[0] * loss_context,
            'loss_inner': w[1] * loss_inner
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)

class ECD_NCD(ECD):
    default_cfg = {
        'dnn_units': [512, 256],
        'dropout_rate': 0.5,
        'activation': 'sigmoid',
        'disc_scale': 10,
        'loss_weight': [1, 1],
    }
    def build_model(self):
        super().build_model()
        self.student_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.k_difficulty = nn.Embedding(self.n_item, self.n_cpt)
        self.e_difficulty = nn.Embedding(self.n_item, 1)
        self.pd_net = PosMLP(
            input_dim=self.n_cpt, output_dim=1, activation=self.modeltpl_cfg['activation'],
            dnn_units=self.modeltpl_cfg['dnn_units'], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )

    def forward(self, stu_id, exer_id):
        items_Q_mat = self.Q_mat[exer_id]
        stus_qqq = self.QQQ_list[stu_id]

        stu_emb = self.student_emb(stu_id)
        theta_inner = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_difficulty = torch.sigmoid(self.e_difficulty(exer_id)) * self.modeltpl_cfg['disc_scale']
        # prednet
        input_knowledge_point = items_Q_mat
        theta_context, dt_w = super().forward(stu_id, stus_qqq)
        theta_all = theta_context * dt_w + theta_inner * (1 - dt_w)

        pre_all = self.pd_net(e_difficulty * (theta_all - k_difficulty) * input_knowledge_point).sigmoid()
        pre_context = self.pd_net(e_difficulty * (theta_context - k_difficulty) * input_knowledge_point).sigmoid()

        input_x = e_difficulty * (theta_inner - k_difficulty) * input_knowledge_point
        pre_inner = self.pd_net(input_x).sigmoid()
        if self.training:
            return pre_all, pre_context, pre_inner
        else:
            return pre_all

    @torch.no_grad()
    def predict(self, stu_id, exer_id,  **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd_all, pd_context, pd_inner = self(stu_id, exer_id)
        pd_all, pd_context, pd_inner = pd_all.flatten(), pd_context.flatten(), pd_inner.flatten()
        loss_all = F.binary_cross_entropy(input=pd_all, target=label)
        loss_context = F.binary_cross_entropy(input=pd_context, target=label)
        loss_inner = F.binary_cross_entropy(input=pd_inner, target=label)
        w = self.modeltpl_cfg['loss_weight']
        # loss = loss_all+w[0]*loss_context+w[1]*loss_inner
        return {
            'loss_main': loss_all,
            'loss_context': w[0]*loss_context,
            'loss_inner': w[1]*loss_inner
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
