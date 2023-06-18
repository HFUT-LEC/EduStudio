import numpy as np
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
from ..utils.components import PosMLP
import torch
import torch.nn.functional as F


class MGCD(GDBaseModel):
    default_cfg = {
        'prednet_len1': 128,
        'prednet_len2': 64,
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def add_extra_data(self, inter_student, df_G):
        # extra_dict['inter_student'], extra_dict['df_G']
        self.stu_n = torch.max(inter_student['stu_id']).item()
        self.df_G = {}
        for index, row in df_G.iterrows():
            self.df_G[row['group_id']] = row['stu_seq']
            max_stu_id = max(row['stu_seq'])
            if max_stu_id > self.stu_n:
                self.stu_n = max_stu_id
        self.stu_n = self.stu_n + 1
        self.stu2exe = {}
        self.stu_exe2label = {}
        for i in range(len(inter_student['stu_id'])):
            stu_id = int(inter_student['stu_id'][i])
            exer_id = int(inter_student['exer_id'][i])
            label = float(inter_student['label'][i])
            if stu_id not in self.stu2exe:
                self.stu2exe[stu_id] = [exer_id]
            else:
                self.stu2exe[stu_id].append(exer_id)
            self.stu_exe2label[(stu_id, exer_id)] = label

    def build_cfg(self):
        self.group_n = self.datafmt_cfg['dt_info']['group_count']
        self.exer_n = self.datafmt_cfg['dt_info']['exer_count']
        self.knowledge_dim = self.datafmt_cfg['dt_info']['cpt_count']
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1 = self.model_cfg['prednet_len1']
        self.prednet_len2 = self.model_cfg['prednet_len2']
        self.num_hidden = self.stu_dim


    def build_model(self):
        self.R = nn.Embedding(self.stu_n, self.stu_dim)
        # h
        self.h = nn.Parameter(torch.randn(self.stu_dim, 1, requires_grad=True))
        # W_k
        self.W_k = nn.Parameter(torch.randn(self.num_hidden, self.stu_dim, requires_grad=True))
        # W_q
        self.W_q = nn.Parameter(torch.randn(self.num_hidden, self.stu_dim, requires_grad=True))
        # W_c
        self.W_c = nn.Embedding(self.group_n, self.stu_dim)

        # A
        self.A = nn.Parameter(torch.randn(self.stu_dim, self.stu_dim, requires_grad=True))
        # B
        self.B = nn.Embedding(self.exer_n, self.knowledge_dim)
        # D
        self.D = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

    def _init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, Q_mat, **kwargs):  # 这里应该得到group对exercise作答正确的概率
        # before prednet
        self.apply_clipper()

        """
        :param group_id: LongTensor
        :param group_member: list of LongTensor
        :param exer_ids: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors  batch_size*knowledge_dim
        :return: FloatTensor, the probabilities of answering correctly
        """
        group_id = stu_id  # 传来的stu_id其实是group_id
        group_member = []
        for i in range(len(group_id)):
            group_member.append(self.df_G[int(group_id[i])])
        kn_emb = Q_mat[exer_id]

        group_emb_list = []
        for i in range(len(group_member)):
            stu_emb = self.R(torch.tensor(group_member[i]))
            ci = self.W_c(group_id[i])
            a = torch.matmul(stu_emb, self.W_k)
            b = torch.mv(self.W_q, ci)
            oj = torch.tanh(a + b)
            oj = torch.matmul(self.h.T, oj.T)
            oj = torch.softmax(oj, dim=1)
            group_emb = torch.matmul(oj, stu_emb)
            group_emb_list.append(group_emb)

        group_emb = torch.cat(group_emb_list)

        group_emb = torch.sigmoid(torch.matmul(group_emb, self.A))
        k_difficulty = torch.sigmoid(self.B(exer_id))
        e_discrimination = torch.sigmoid(self.D(exer_id)) * 10

        input_x = e_discrimination * (group_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    @torch.no_grad()
    def predict(self, stu_id, exer_id, Q_mat, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id, Q_mat).flatten(),
        }

    def get_main_loss(self, **kwargs):
        # 这里的loss跟论文中loss对应
        group_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        Q_mat = kwargs['Q_mat']
        pd = self(group_id, exer_id, Q_mat).flatten()
        loss_mse = nn.MSELoss(reduction = 'mean')
        loss_group = loss_mse(pd, label)

        stu_ids = []
        for i in range(len(group_id)):
            stu_ids = stu_ids + self.df_G[int(group_id[i])]
        stu_id = []
        exer_stu = []
        label_stu = []
        for stu in stu_ids:
            if stu in self.stu2exe:
                for exer in self.stu2exe[stu]:
                    stu_id.append(stu)
                    exer_stu.append(exer)
                    label_stu.append(self.stu_exe2label[(stu, exer)])
        stu_id = torch.tensor(stu_id)
        exer_id = torch.tensor(exer_stu)
        label_stu = torch.unsqueeze(torch.tensor(label_stu), dim=1)

        kn_emb = Q_mat[exer_id]
        stu_emb = self.R(stu_id)
        stu_emb = torch.sigmoid(torch.matmul(stu_emb, self.A))
        k_difficulty = torch.sigmoid(self.B(exer_id))
        e_discrimination = torch.sigmoid(self.D(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        loss_stu = F.binary_cross_entropy(input=output, target=label_stu)

        return {
            'loss_main': loss_group + loss_stu
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.W_c(stu_id)
        else:
            return self.W_c.weight


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
