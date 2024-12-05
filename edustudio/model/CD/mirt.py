r"""
MIRT
##########################################

Reference:
    Mark D Reckase et al. "Multidimensional item response theory models". Springer, 2009.

Reference Code:
    https://github.com/bigdata-ustc/EduCDM/tree/main/EduCDM/MIRT

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..gd_basemodel import GDBaseModel


class MIRT(GDBaseModel):
    """
        第一种: fix_a = True, fix_c = True
        第二种: fix_a = False, fix_c = True
        第三种: fix_a = False, fix_c = False
    """
    default_cfg = {
        "a_range": -1.0, # disc range
        "emb_dim": 32
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        if self.modeltpl_cfg['a_range'] is not None and self.modeltpl_cfg['a_range']  < 0: self.modeltpl_cfg['a_range'] = None

        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.emb_dim = self.modeltpl_cfg['emb_dim']

    def build_model(self):
        self.theta = nn.Embedding(self.n_user, self.emb_dim) # student ability
        self.a = nn.Embedding(self.n_item, self.emb_dim) # exer discrimination
        self.b = nn.Embedding(self.n_item, 1) # exer intercept term

    def forward(self, stu_id, exer_id, **kwargs):
        theta = self.theta(stu_id)
        a = self.a(exer_id)
        b = self.b(exer_id).flatten()

        if self.modeltpl_cfg['a_range'] is not None:
            a = self.modeltpl_cfg['a_range'] * torch.sigmoid(a)
        else:
            a = F.softplus(a) # 让区分度大于0，保持单调性假设
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The diff_range or a_range is too large.')
        return self.irf(theta, a, b)

    @staticmethod
    def irf(theta, a, b):
        return 1 / (1 + torch.exp(- torch.sum(torch.multiply(a, theta), axis=-1) + b)) # 为何sum前要取负号

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
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }
