r"""
CNCD-Q
##########################################

Reference:
    Fei Wang et al. "Neural Cognitive Diagnosis for Intelligent Education Systems" in AAAI 2020.

Reference Code:
    https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD
    https://github.com/LegionKing/NeuralCDM_plus

"""
import numpy as np
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
from ..utils.components import PosMLP
import torch
import torch.nn.functional as F


class CNCD_Q(GDBaseModel):
    r"""
    CNCD-Q

    default_cfg:
       'dnn_units': [512, 256]  # dimension list of hidden layer in prediction layer
       'dropout_rate': 0.5      # dropout rate
       'activation': 'sigmoid'  # activation function in prediction layer
       'disc_scale': 10         # discrimination scale
    """
    default_cfg = {
        'dnn_units': [512, 256],
        'dropout_rate': 0.5,
        'activation': 'sigmoid',
        'disc_scale': 10
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def build_model(self):
        # prediction sub-net
        self.student_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.k_difficulty = nn.Embedding(self.n_item, self.n_cpt)
        self.e_difficulty = nn.Embedding(self.n_item, 1)
        self.pd_net = PosMLP(
            input_dim=self.n_cpt, output_dim=1, activation=self.modeltpl_cfg['activation'],
            dnn_units=self.modeltpl_cfg['dnn_units'], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )
        self.e_k_prob = nn.Embedding(self.n_item, self.n_cpt)

    def forward(self, stu_id, exer_id, Q_mat, **kwargs):
        # before prednet
        users = stu_id
        items = exer_id
        items_Q_mat = Q_mat[items]
        stu_emb = self.student_emb(users)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(items))
        e_difficulty = torch.sigmoid(self.e_difficulty(items)) * self.modeltpl_cfg['disc_scale']
        # prednet
        e_k_prob = self.e_k_prob(items)
        e_k_prob_2 = F.sigmoid(e_k_prob)  # knowledge relevancy vectors of the exercises
        input_x = e_difficulty * (stat_emb - k_difficulty) * (items_Q_mat * e_k_prob_2)
        pd = self.pd_net(input_x).sigmoid()
        if self.training:
            return pd, e_k_prob
        else:
            return pd

    @torch.no_grad()
    def predict(self, stu_id, exer_id, Q_mat, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id, Q_mat).flatten(),
        }

    def get_main_loss(self, **kwargs):
        normal_mean, normal_C = 0, 2
        means = torch.ones(self.n_cpt) * normal_mean  # the mean of the multidimensional gaussian distribution
        means.require_grad = False
        means = means.to(self.traintpl_cfg['device'])
        C = torch.ones(self.n_cpt) * normal_C  # the diagonal of the covariance matrix
        C.require_grad = False
        C = C.to(self.traintpl_cfg['device'])
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        Q_mat = kwargs['Q_mat']
        knowledge_pairs_kw = kwargs['knowledge_pairs']
        # kn_tops = kwargs["kn_tops"]
        # kn_tags = kwargs["kn_tags"]
        pd, exer_knowledge_prob= self(stu_id, exer_id, Q_mat=Q_mat)
        pd = pd.flatten()
        loss_1 = F.binary_cross_entropy(input=pd, target=label)
        loss_2 = 0
        knowledge_pairs =[]
        for i in exer_id.cpu().numpy():
            knowledge_pairs.append(knowledge_pairs_kw[i])
        for pair_i in range(len(knowledge_pairs)):#batch_size
            kn_tags, kn_topks = knowledge_pairs[pair_i]
            kn_tags, kn_topks = np.array(kn_tags) - 0, np.array(kn_topks) - 0  # 转成从0开始
            # if -1 in kn_tags or -1 in kn_topks:
            #     print(kn_tags)
            kn_tag_n = len(kn_tags)
            kn_tag_tensor = exer_knowledge_prob[pair_i][kn_tags].view(-1, 1)
            kn_prob_tensor = exer_knowledge_prob[pair_i][kn_topks].repeat(kn_tag_n, 1)
            loss_2 = loss_2 - (torch.log(torch.sigmoid((kn_tag_tensor - kn_prob_tensor) * 0.1))).sum()
        for kn_prob in exer_knowledge_prob:
            a = kn_prob - means
            loss_2 = loss_2 + 0.5 * (a * a / C).sum()


        return {
            'loss_1': loss_1,
            'loss_2': loss_2
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.student_emb(stu_id)
        else:
            return self.student_emb.weight
