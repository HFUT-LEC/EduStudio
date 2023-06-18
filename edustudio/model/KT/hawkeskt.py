# -*- coding: UTF-8 -*-

import numpy as np
from ..gd_basemodel import GDBaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class HawkesKT(GDBaseModel):
    default_cfg = {
        'dim_s': 50,  # 序列长度
        'emb_size': 64,
        'time_log': 5,  # Log base of time intervals.
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.problem_num = self.datafmt_cfg['dt_info']['exer_count']  # 习题数量
        self.skill_num = self.datafmt_cfg['dt_info']['cpt_count']  # 知识点数量
        self.emb_size = self.model_cfg['emb_size']
        self.time_log = self.model_cfg['time_log']

    def build_model(self):
        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)  # 对应论文中的PA吧
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

    def _init_params(self):
        #  super()._init_params()
        torch.nn.init.normal_(self.problem_base.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.skill_base.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.alpha_inter_embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.alpha_skill_embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.beta_inter_embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.beta_skill_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, exer_seq, time_lag_seq, cpt_unfold_seq, **kwargs):
        skills = cpt_unfold_seq     # [batch_size, seq_len] 一个习题对应一个知识点
        problems = exer_seq  # [batch_size, seq_len] batch_size个学生的序列
        # time = [i for i in range(time_lag_seq.shape[1])]
        # times = torch.Tensor([time for i in range(time_lag_seq.shape[0])])
        times = time_lag_seq - time_lag_seq[:,[0]]        # [batch_size, seq_len]

        mask_labels = kwargs['mask_seq'].long()
        inters = skills + mask_labels * self.skill_num

        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(skills)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)  
        # torch.clamp把betas+1所有元素变为0~10，即betas+1小于0的元素变为0，betas+1大于10的元素变为10，其余不变

        delta_t = (times[:, :, None] - times[:, None, :]).abs().double()  # 得到不同时间步的时间的绝对值
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)  # 论文(4)式的cross_effects

        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.cuda() if self.device != 'cpu' else mask
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(skills).squeeze(dim=-1)

        prediction = (problem_bias + skill_bias + sum_t).sigmoid()
        return prediction

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def get_main_loss(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd.double(), target=y_gt.double()
        )
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
