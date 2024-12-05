r"""
NCDM
##########################################

Reference:
    Fei Wang et al. "Neural Cognitive Diagnosis for Intelligent Education Systems" in AAAI 2020.

Reference Code:
    https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD
    https://github.com/bigdata-ustc/EduCDM/tree/main/EduCDM/NCDM

"""

from typing import Any
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
from ..utils.components import PosMLP
import torch
import torch.nn.functional as F


class NCDM(GDBaseModel):
    r"""
    NCDM

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

    def add_extra_data(self, **kwargs):
        self.Q_mat = kwargs['Q_mat'].to(self.device)

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

    def forward(self, stu_id, exer_id, **kwargs):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_difficulty = torch.sigmoid(self.e_difficulty(exer_id)) * self.modeltpl_cfg['disc_scale']
        # prednet
        input_knowledge_point = self.Q_mat[exer_id]
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        pd = self.pd_net(input_x).sigmoid()
        return pd

    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

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
    
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.student_emb(stu_id)
        else:
            return self.student_emb.weight
