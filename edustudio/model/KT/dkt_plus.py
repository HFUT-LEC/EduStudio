from .dkt import DKT
import torch.nn as nn
import torch
import torch.nn.functional as F


class DKT_plus(DKT):
    default_cfg = {
        'lambda_r': 0.01,
        'lambda_w1': 0.003,
        'lambda_w2': 3.0,
        'reg_all_KCs': True
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def get_main_loss(self, **kwargs):
        q = kwargs['exer_seq'][:, :-1].unsqueeze(dim=-1)
        q_shft = kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1)

        y = self(**kwargs)
        pred = y[:, :-1].gather(index=q, dim=2).squeeze(dim=-1)
        pred_shft = y[:, :-1].gather(index=q_shft, dim=2).squeeze(dim=-1)

        y_curr = pred[kwargs['mask_seq'][:, :-1] == 1]
        y_next = pred_shft[kwargs['mask_seq'][:, 1:] == 1]

        gt_curr =  kwargs['label_seq'][:, :-1][kwargs['mask_seq'][:, :-1] == 1]
        gt_next = kwargs['label_seq'][:, 1:][kwargs['mask_seq'][:, 1:] == 1]

        loss_main = F.binary_cross_entropy(input=y_next, target=gt_next)
        loss_r = self.modeltpl_cfg['lambda_r'] * F.binary_cross_entropy(input=y_curr, target=gt_curr)

        if self.modeltpl_cfg['reg_all_KCs']:
            diff = y[:, 1:] - y[:, :-1]
        else:
            diff = (pred_shft - pred)[kwargs['mask_seq'][:, 1:] == 1]
        loss_w1 = torch.norm(diff, 1) / len(diff)
        loss_w1 = self.modeltpl_cfg['lambda_w1'] * loss_w1 / self.n_item
        loss_w2 = torch.norm(diff, 2) / len(diff)
        loss_w2 = self.modeltpl_cfg['lambda_w2'] * loss_w2 / self.n_item

        return {
            'loss_main': loss_main,
            'loss_r': loss_r,
            'loss_w1': loss_w1,
            'loss_w2': loss_w2
        }
