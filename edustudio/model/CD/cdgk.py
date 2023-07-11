r"""
CDGK
##################################
Reference:
    Haiping Ma et al. "Using Knowledge Concept Aggregation towards Accurate Cognitive Diagnosis." in CIKM 2021.
"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
from ..utils.components import PosLinear
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
from edustudio.utils.common import tensor2npy
import numpy as np
from typing import Dict


class CDGK_META(nn.Module):
    def __init__(self, stu_count, exer_count, cpt_count):
        super().__init__()
        self.n_user = stu_count
        self.n_item = exer_count
        self.n_cpt = cpt_count
        self.build_model()

    def build_model(self):
        self.stu_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.exer_emb = nn.Embedding(self.n_item, self.n_cpt)
        self.exer_disc = nn.Embedding(self.n_item, 1)
        self.guess_stu_emb = nn.Embedding(self.n_user, 1)
        self.guess_cpt_emb = nn.Embedding(self.n_cpt, 1)

        self.pred_layer = nn.Sequential(
            PosLinear(self.n_cpt, 1),
            nn.Sigmoid()
        )

        self.guess_pred_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, stu_id, exer_id, **kwargs):
        items_Q_mat = self.Q_mat[exer_id]
        stu_emb = self.stu_emb(stu_id).sigmoid()
        k_difficulty = self.exer_emb(exer_id).sigmoid()
        e_difficulty = self.exer_disc(exer_id).sigmoid()

        input_x = e_difficulty * (stu_emb - k_difficulty) * items_Q_mat
        pd1 = self.pred_layer(input_x)

        pd2 = self.guess_pred_layer(
            torch.cat(
                [self.guess_stu_emb(stu_id), torch.mm(items_Q_mat.float(), self.guess_cpt_emb.weight)], dim=1
            ).sum(dim=1, keepdim=True)
        )
        return pd2  + (1 - pd2) * pd1

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
            return self.stu_emb(stu_id)
        else:
            return self.stu_emb.weight


class CDGK_SINGLE(GDBaseModel):
    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def build_model(self):
        self.cdgk = CDGK_META(
            stu_count=self.n_user,
            exer_count=self.n_item,
            cpt_count=self.n_cpt
        )
        self.cdgk.Q_mat = self.Q_mat

    def add_extra_data(self, **kwargs):
        super().add_extra_data(**kwargs)
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    @torch.no_grad()
    def predict(self, **kwargs):
        return self.cdgk.predict(**kwargs)

    def get_main_loss(self, **kwargs):
        return self.cdgk.get_main_loss(**kwargs)

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    def get_stu_status(self, stu_id=None):
        return self.cdgk.get_stu_status(stu_id=stu_id)


class CDGK_MULTI(GDBaseModel):
    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.n_cpt_group = self.datatpl_cfg['dt_info']['n_cpt_group']


    def build_model(self):
        self.cdgk_list = nn.ModuleList(
            [
                CDGK_META(
                    stu_count=self.n_user, exer_count=self.n_item, cpt_count=self.n_cpt
                ) for _ in range(self.n_cpt_group)
            ]
        )
        for cdgk_meta in self.cdgk_list:
            cdgk_meta.Q_mat = self.Q_mat

    def add_extra_data(self, **kwargs):
        super().add_extra_data(**kwargs)
        self.Q_mat = kwargs['Q_mat'].to(self.device)

        self.gid2exers: Dict[int, torch.LongTensor] = {
            k: torch.from_numpy(v).long().to(self.device) 
            for k,v in kwargs['gid2exers'].items()
        }

        self.n_group_of_cpt: torch.LongTensor = torch.from_numpy(
            kwargs['n_group_of_cpt']
        ).long().to(self.device) 
        
    def forward(self, stu_id, exer_id, **kwargs):
        g_pd_list, g_order_list = [], []
        for gid, sub_cdgk in enumerate(self.cdgk_list):
            g_flag = torch.isin(exer_id, self.gid2exers[gid])
            if len(g_flag) == 0: continue

            g_stu_id = stu_id[g_flag]
            g_exer_id = exer_id[g_flag]
            g_pd = sub_cdgk(g_stu_id, g_exer_id)
            
            g_pd_list.append(g_pd)
            g_order_list.append(torch.argwhere(g_flag).flatten())

        order = torch.concat(g_order_list)
        pd = torch.concat(g_pd_list)

        pd, _ = self.groupby_mean(pd, labels=order, device=self.device)
        return pd.flatten()
    
    @torch.no_grad()
    def predict(self, **kwargs):
        return {"y_pd": self(**kwargs)}

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
            stu_emb_mat = torch.stack([sub_cdgk.stu_emb(stu_id) for sub_cdgk in self.cdgk_list], dim=2)
        else:
            stu_emb_mat = torch.stack([sub_cdgk.stu_emb.weight for sub_cdgk in self.cdgk_list], dim=2)
        
        return stu_emb_mat.sum(dim=-1) / self.n_group_of_cpt

    # reference: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/9
    @staticmethod
    def groupby_mean(value:torch.Tensor, labels:torch.LongTensor, device="cuda:0"):
        """Group-wise average for (sparse) grouped tensors
        
        Args:
            value (torch.Tensor): values to average (# samples, latent dimension)
            labels (torch.LongTensor): labels for embedding parameters (# samples,)
        
        Returns: 
            result (torch.Tensor): (# unique labels, latent dimension)
            new_labels (torch.LongTensor): (# unique labels,)
            
        Examples:
            >>> samples = torch.Tensor([
                                [0.15, 0.15, 0.15],    #-> group / class 1
                                [0.2, 0.2, 0.2],    #-> group / class 3
                                [0.4, 0.4, 0.4],    #-> group / class 3
                                [0.0, 0.0, 0.0]     #-> group / class 0
                        ])
            >>> labels = torch.LongTensor([1, 5, 5, 0])
            >>> result, new_labels = groupby_mean(samples, labels)
            
            >>> result
            tensor([[0.0000, 0.0000, 0.0000],
                [0.1500, 0.1500, 0.1500],
                [0.3000, 0.3000, 0.3000]])
                
            >>> new_labels
            tensor([0, 1, 5])
        """
        uniques = labels.unique().tolist()
        labels = labels.tolist()

        key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
        val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}
        
        labels = torch.LongTensor(list(map(key_val.get, labels)))
        
        labels = labels.view(labels.size(0), 1).expand(-1, value.size(1)).to(device)
        
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        result = torch.zeros_like(unique_labels, dtype=torch.float, device=device).scatter_add_(0, labels, value)
        result = result / labels_count.float().unsqueeze(1)
        new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
        return result, new_labels
