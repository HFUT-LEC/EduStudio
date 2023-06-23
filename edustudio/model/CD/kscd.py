r"""
KSCD
##########################################

Reference:
    Haiping Ma et al. "Knowledge-Sensed Cognitive Diagnosis for Intelligent Education Platforms" in CIKM 2022.

Reference Code:
    https://github.com/BIMK/Intelligent-Education/tree/main/KSCD_Code_F

"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..utils.components import PosMLP, PosLinear


class KSCD(GDBaseModel):
    default_cfg = {
        'emb_dim': 20,
        'dropout_rate': 0.5,
        'interaction_type': 'kscd', # ['kscd', 'ncdm']
        'interaction_type_ncdm': {
            'dnn_units': [512, 256],
            'dropout_rate': 0.5,
            'activation': 'sigmoid',
            'disc_scale': 10
        },
        'interaction_type_kscd': {
            'dropout_rate': 0.0,
        }
    }
    def __init__(self, cfg):
        super().__init__(cfg)    

    def add_extra_data(self, **kwargs):
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    def build_cfg(self):
        self.stu_count = self.datatpl_cfg['dt_info']['stu_count']
        self.exer_count = self.datatpl_cfg['dt_info']['exer_count']
        self.cpt_count = self.datatpl_cfg['dt_info']['cpt_count']
    
        self.lowdim = self.modeltpl_cfg['emb_dim']
        self.interaction_type = self.modeltpl_cfg['interaction_type']

    def build_model(self):
        self.stu_emb = nn.Embedding(self.stu_count, self.lowdim)
        self.cpt_emb = nn.Embedding(self.cpt_count, self.lowdim)
        self.exer_emb = nn.Embedding(self.exer_count, self.lowdim)

        if self.modeltpl_cfg['interaction_type'] == 'ncdm':
            # self.exer_disc_infer = nn.Linear(self.lowdim, 1)
            self.layer1 = nn.Linear(self.lowdim, 1)
            self.pd_net = PosMLP(
                input_dim=self.cpt_count, output_dim=1, 
                activation=self.modeltpl_cfg['interaction_type_ncdm']['activation'],
                dnn_units=self.modeltpl_cfg['interaction_type_ncdm']['dnn_units'], 
                dropout_rate=self.modeltpl_cfg['interaction_type_ncdm']['dropout_rate']
            )
        elif self.modeltpl_cfg['interaction_type'] == 'kscd':
            self.prednet_full1 = PosLinear(self.cpt_count + self.lowdim, self.cpt_count, bias=False)
            self.drop_1 = nn.Dropout(p=self.modeltpl_cfg['interaction_type_kscd']['dropout_rate'])
            self.prednet_full2 = PosLinear(self.cpt_count + self.lowdim, self.cpt_count, bias=False)
            self.drop_2 = nn.Dropout(p=self.modeltpl_cfg['interaction_type_kscd']['dropout_rate'])
            self.prednet_full3 = PosLinear(1 * self.cpt_count, 1)
        else:
            raise ValueError(f"unknown interaction_type: {self.modeltpl_cfg['interaction_type']}")

    def forward(self, stu_id, exer_id, **kwargs):
        
        stu_emb = self.stu_emb(stu_id)
        exer_emb = self.exer_emb(exer_id)
        exer_q_mat = self.Q_mat[exer_id]

        stu_ability = torch.mm(stu_emb, self.cpt_emb.weight.T).sigmoid()
        exer_diff = torch.mm(exer_emb, self.cpt_emb.weight.T).sigmoid()
        
        if self.modeltpl_cfg['interaction_type'] == 'ncdm':
            # exer_disc = self.exer_disc_infer(exer_emb).sigmoid() * self.modeltpl_cfg['interaction_type_ncdm']['disc_scale']
            exer_disc = torch.sigmoid(self.layer1(exer_emb)) * self.modeltpl_cfg['interaction_type_ncdm']['disc_scale']
            input_x = exer_disc * (stu_ability - exer_diff) * exer_q_mat
            y_pd = self.pd_net(input_x).sigmoid()
        elif self.modeltpl_cfg['interaction_type'] == 'kscd':
            batch_stu_vector = stu_ability.repeat(1, self.cpt_count).reshape(stu_ability.shape[0], self.cpt_count, stu_ability.shape[1])
            batch_exer_vector = exer_diff.repeat(1, self.cpt_count).reshape(exer_diff.shape[0], self.cpt_count, exer_diff.shape[1])

            kn_vector = self.cpt_emb.weight.repeat(stu_ability.shape[0], 1).reshape(stu_ability.shape[0], self.cpt_count, self.lowdim)

            # CD
            preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
            diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
            o = torch.sigmoid(self.prednet_full3(preference - diff))

            sum_out = torch.sum(o * exer_q_mat.unsqueeze(2), dim=1)
            count_of_concept = torch.sum(exer_q_mat, dim=1).unsqueeze(1)
            y_pd = sum_out / count_of_concept
        else:
            raise ValueError(f"unknown interaction_type: {self.modeltpl_cfg['interaction_type']}")

        return y_pd

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd = self(stu_id=stu_id, exer_id=exer_id).flatten()
        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id=stu_id, exer_id=exer_id).flatten(),
        }

    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            stu_emb = self.stu_emb(stu_id)
        else:
            stu_emb = self.stu_emb.weight
        stu_ability = torch.mm(stu_emb, self.cpt_emb.weight.T).sigmoid()
        return stu_ability
