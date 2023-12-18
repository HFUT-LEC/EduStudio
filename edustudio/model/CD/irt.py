r"""
IRT
##########################################

Reference Code:
    https://github.com/bigdata-ustc/EduCDM/tree/main/EduCDM/IRT

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..gd_basemodel import GDBaseModel


class IRT(GDBaseModel):
    r"""
    IRT
    """
    default_cfg = {
        "a_range": -1.0, # disc range
        "diff_range": -1.0, # diff range
        "fix_a": False,
        "fix_c": True,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        if self.modeltpl_cfg['a_range'] is not None and self.modeltpl_cfg['a_range']  < 0: self.modeltpl_cfg['a_range'] = None
        if self.modeltpl_cfg['diff_range'] is not None and self.modeltpl_cfg['diff_range'] < 0: self.modeltpl_cfg['diff_range'] = None

        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']

        # 确保c固定时，a一定不能固定
        if self.modeltpl_cfg['fix_c'] is False: assert self.modeltpl_cfg['fix_a'] is False

    def build_model(self):
        self.theta = nn.Embedding(self.n_user, 1) # student ability
        self.a = 0.0 if self.modeltpl_cfg['fix_a'] else nn.Embedding(self.n_item, 1) # exer discrimination
        self.b = nn.Embedding(self.n_item, 1) # exer difficulty
        self.c = 0.0 if self.modeltpl_cfg['fix_c'] else nn.Embedding(self.n_item, 1)

    def forward(self, stu_id, exer_id, **kwargs):
        theta = self.theta(stu_id)
        a = self.a(exer_id)
        b = self.b(exer_id)
        c = self.c if self.modeltpl_cfg['fix_c'] else self.c(exer_id).sigmoid()

        if self.modeltpl_cfg['diff_range'] is not None:
            b = self.modeltpl_cfg['diff_range'] * (torch.sigmoid(b) - 0.5)
        if self.modeltpl_cfg['a_range'] is not None:
            a = self.modeltpl_cfg['a_range'] * torch.sigmoid(a)
        else:
            a = F.softplus(a) # 让区分度大于0，保持单调性假设
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The diff_range or a_range is too large.')
        return self.irf(theta, a, b, c)

    @staticmethod
    def irf(theta, a, b, c, D=1.702):
        return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)))

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

