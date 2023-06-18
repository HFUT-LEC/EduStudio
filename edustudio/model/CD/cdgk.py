from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
from ..utils.components import PosLinear
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
from edustudio.utils.common import tensor2npy
import numpy as np

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

    def forward(self, stu_id, exer_id, Q_mat, **kwargs):
        items_Q_mat = Q_mat[exer_id]
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
    def predict(self, stu_id, exer_id, Q_mat, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id, Q_mat).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        Q_mat = kwargs['Q_mat']
        pd = self(stu_id, exer_id, Q_mat).flatten()
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
        self.n_user = self.datafmt_cfg['dt_info']['stu_count']
        self.n_item = self.datafmt_cfg['dt_info']['exer_count']
        self.n_cpt = self.datafmt_cfg['dt_info']['cpt_count']

    def build_model(self):
        self.cdgk = CDGK_META(
            stu_count=self.n_user,
            exer_count=self.n_item,
            cpt_count=self.n_cpt
        )

    @torch.no_grad()
    def predict(self, **kwargs):
        return self.cdgk.predict(**kwargs)

    def get_main_loss(self, **kwargs):
        self.cdgk.get_main_loss(**kwargs)

    def get_loss_dict(self, **kwargs):
        return self.cdgk.get_main_loss(**kwargs)
    
    def get_stu_status(self, stu_id=None):
        return self.cdgk.get_stu_status(stu_id=stu_id)


class CDGK_MULTI(GDBaseModel):
    def build_cfg(self):
        self.n_user = self.datafmt_cfg['dt_info']['stu_count']
        self.n_item = self.datafmt_cfg['dt_info']['exer_count']
        self.n_cpt = self.datafmt_cfg['dt_info']['cpt_count']

    
    def add_extra_data(self, **kwargs):
        # 1. 知识点与图对应情况
        # 2. 知识点分组情况
        self.cpt2group = {}
        self.df_cpt2group = None
        self.cpt_group_num_list = [10, 20, 30]
        self.sub_model_num = len(self.cpt_group_num_list)


    def build_model(self):
        self.cdgk_list = nn.ModuleList(
            [
                CDGK_META(
                    stu_count=self.n_user, exer_count=self.n_item, cpt_count=cpt_count
                ) for cpt_count in self.cpt_group_num_list
            ]
        )


    def forward(self, stu_id, exer_id, Q_mat, **kwargs):
        batch_Q_mat = Q_mat[exer_id]
        df = pd.DataFrame(tensor2npy(torch.argwhere(batch_Q_mat == 1)), columns=['idx','cpt_id'])
        df = df.merge(self.df_cpt2group, on='cpt_id', how='left')
        group2idx = df[['idx','group_id']].groupby('group_id').agg(
            lambda x: torch.from_numpy(np.array(list(x))).to(self.device)
        )['idx'].to_dict()

        pd_dict = {}
        for mid in range(self.sub_model_num):
            if mid not in group2idx: continue
            tmp_stu_id = stu_id[group2idx[mid]]
            tmp_exer_id = exer_id[group2idx[mid]]
            pd = self.cdgk_list[mid](stu_id=tmp_stu_id, exer_id=tmp_exer_id, Q_mat=Q_mat[:, self.cpt2group[mid]])
            pd_dict[mid] = pd
        
        # 合并预测值，割点取均值
        


