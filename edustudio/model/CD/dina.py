r"""
DINA
##########################################

Reference:
    Jimmy De La Torre. Dina model and parameter estimation: A didactic. Journal of educational and behavioral statistics, 34(1):115–130, 2009

Reference Code:
    https://github.com/bigdata-ustc/EduCDM/blob/main/EduCDM/DINA/

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..gd_basemodel import GDBaseModel
import torch.autograd as autograd


class DINA(GDBaseModel):
    default_cfg = {
        "step": 0,
        "max_step": 1000,
        "max_slip": 0.4,
        "max_guess": 0.4,
    }
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def add_extra_data(self, **kwargs):
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

        self.step = 0
        self.max_step = self.modeltpl_cfg['max_step']
        self.max_slip =  self.modeltpl_cfg['max_slip']
        self.max_guess =  self.modeltpl_cfg['max_guess']
        self.emb_dim = self.n_cpt

    def build_model(self):
        self.guess = nn.Embedding(self.n_item, 1)
        self.slip = nn.Embedding(self.n_item, 1)
        self.theta = nn.Embedding(self.n_user, self.emb_dim)

    def forward(self, stu_id, exer_id, **kwargs):
        items_Q_mat = self.Q_mat[exer_id]
        theta = self.theta(stu_id)
        slip = torch.squeeze(torch.sigmoid(self.slip(exer_id)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(exer_id)) * self.max_guess)

        knowledge = items_Q_mat
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)
    
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
                
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.theta(stu_id)
        else:
            return self.theta.weight

class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINA(DINA):
    default_cfg = {
        "max_slip": 0.4,
        "max_guess": 0.4,
    }
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sign = StraightThroughEstimator()

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

        self.max_slip =  self.modeltpl_cfg['max_slip']
        self.max_guess =  self.modeltpl_cfg['max_guess']
        self.emb_dim = self.n_cpt

    def forward(self, stu_id, exer_id):
        theta = self.theta(stu_id)
        theta = self.sign(self.theta(stu_id))
        knowledge = self.Q_mat[exer_id]
        slip = torch.squeeze(torch.sigmoid(self.slip(exer_id)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(exer_id)) * self.max_guess)
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)
