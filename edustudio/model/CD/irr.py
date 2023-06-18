from ..basemodel import BaseProxyModel
import torch
import torch.nn as nn


class PairSCELoss(nn.Module):
    def __init__(self):
        super(PairSCELoss, self).__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, pred1, pred2, sign=1, *args):
        """
        sign is either 1 or -1
        could be seen as predicting the sign based on the pred1 and pred2
        1: pred1 should be greater than pred2
        -1: otherwise
        """
        pred = torch.stack([pred1, pred2], dim=1)
        return self._loss(pred, ((torch.ones(pred1.shape[0], device=pred.device) - sign) / 2).long())


class IRR(BaseProxyModel):
    default_cfg = {
        "backbone_model_cls": "IRT",
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        super().build_model()
        self.irr_pair_loss = PairSCELoss()

    def get_main_loss(self, **kwargs):
        pair_exer = kwargs['pair_exer']
        pair_pos_stu = kwargs['pair_pos_stu']
        pair_neg_stu = kwargs['pair_neg_stu']
        
        kwargs['exer_id'] = pair_exer
        kwargs['stu_id'] = pair_pos_stu
        pos_pd = self(**kwargs).flatten()
        kwargs['stu_id'] = pair_neg_stu
        neg_pd = self(**kwargs).flatten()

        return {
            'loss_main': self.irr_pair_loss(pos_pd, neg_pd)
        }
