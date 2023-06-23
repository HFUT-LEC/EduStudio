r"""
CDMFKC
##################################
Reference:
    Li et al. "Cognitive Diagnosis Focusing on Knowledge Concepts." in CIKM 2022.
"""
import torch
import torch.nn as nn
from ..utils.components import PosMLP
import torch.nn.functional as F
from ..gd_basemodel import GDBaseModel


class CDMFKC(GDBaseModel):
    """
    dnn_units: dimensions of the middle layers of a multilayer perceptron
    dropout_rate: dropout rate of a multilayer perceptron
    activation: activation function of a multilayer perceptron
    g_impact_a: hyperparameters of the original formula 5
    g_impact_b: hyperparameters of the original formula 5
    """
    default_cfg = {
        'dnn_units': [512, 256],
        'dropout_rate': 0.5,
        'activation': 'sigmoid',
        'g_impact_a': 0.5,
        'g_impact_b': 0.5
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def add_extra_data(self, **kwargs):
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    def build_model(self):
        self.student_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.k_difficulty = nn.Embedding(self.n_item, self.n_cpt)
        self.e_difficulty = nn.Embedding(self.n_item, 1)
        self.k_impact = nn.Embedding(self.n_item, self.n_cpt)
        self.pd_net = PosMLP(
            input_dim=self.n_cpt, output_dim=1, activation=self.modeltpl_cfg['activation'],
            dnn_units=self.modeltpl_cfg['dnn_units'], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )

    def forward(self, stu_id, exer_id, **kwargs):
        # before prednet
        items_Q_mat = self.Q_mat[exer_id]  # Q_mat: exer_num * n_cpt; items_Q_mat: batch_exer_num * n_cpt
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)

        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_difficulty = torch.sigmoid(self.e_difficulty(exer_id))
        h_impact = torch.sigmoid(self.k_impact(exer_id))
        g_impact = torch.sigmoid(self.modeltpl_cfg['g_impact_a'] * h_impact + 
                                 self.modeltpl_cfg['g_impact_b'] * k_difficulty * e_difficulty)
        # k_num = torch.sum(items_Q_mat, dim=1)  #  batch_exer_num
        # avg_impact = torch.multiply(k_num, torch.sum(torch.multiply(items_Q_mat, g_impact), dim=1))
        # k_difficulty_sum = torch.sum(items_Q_mat * k_difficulty, dim=1)
        # stu_stat_sum = torch.sum(stat_emb * items_Q_mat, dim=1)
        # slip = avg_impact * torch.div(k_difficulty_sum, stu_stat_sum)
        # guess = avg_impact * torch.div(stu_stat_sum, k_difficulty_sum)
        # prednet
        input_knowledge_point = items_Q_mat
        input_x = e_difficulty * (stat_emb + g_impact - k_difficulty) * input_knowledge_point

        pd = self.pd_net(input_x).sigmoid()
        # pd = torch.where(pd<0.5, (1 - guess).unsqueeze(1) * pd, pd)
        # pd = torch.where(pd>=0.5, (1 - slip).unsqueeze(1) * pd, pd)
        return pd


    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd = self(stu_id, exer_id).flatten()
        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    @torch.no_grad()
    def predict(self, stu_id, exer_id,  **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }
    
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.student_emb(stu_id)
        else:
            return self.student_emb.weight
