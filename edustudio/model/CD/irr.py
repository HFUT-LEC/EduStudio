r"""
IRR
##################################
Reference:
    Tong et al. "Item Response Ranking for Cognitive Diagnosis." in IJCAI 2021.
"""
from ..basemodel import BaseProxyModel
import torch
import torch.nn as nn


class PairSCELoss(nn.Module):
    """IRR loss function"""
    def __init__(self):
        """IRR loss will use cross entropy"""
        super(PairSCELoss, self).__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, pred1, pred2, sign=1, *args):
        """Get the PairSCELoss

        Args:
            pred1 (torch.Tensor): positive prediction
            pred2 (_type_): negtive prediction
            sign (int, optional): 1: pred1 should be greater than pred2; -1: otherwise. Defaults to 1.

        Returns:
            torch.Tensor: PairSCELoss
        """
        pred = torch.stack([pred1, pred2], dim=1)
        return self._loss(pred, ((torch.ones(pred1.shape[0], device=pred.device) - sign) / 2).long())



class IRR(BaseProxyModel):
    """
    backbone_modeltpl_cls: The backbone model of IRR
    """
    default_cfg = {
        "backbone_modeltpl_cls": "IRT",
    }

    def __init__(self, cfg):
        """Pass parameters from other templates into the model

        Args:
            cfg (UnifyConfig): parameters from other templates
        """
        super().__init__(cfg)

    def build_model(self):
        """Initialize the various components of the model"""
        super().build_model()
        self.irr_pair_loss = PairSCELoss()

    def get_main_loss(self, **kwargs):
        """Get the loss of IRR

        Returns:
            dict: {'loss_main': loss}
        """
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
