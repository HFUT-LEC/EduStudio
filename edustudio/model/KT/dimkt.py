import math
from torch.autograd import Variable
from torch.nn import Embedding, Linear, Sigmoid, Tanh, Dropout

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, constant_

class DIMKT(GDBaseModel):
    default_cfg = {
        'emb_size': 128,
        'num_steps': 199,
        'difficult_levels': 100+2,
        'dropout':0.2
    }

    def build_cfg(self):
        self.n_user = self.datafmt_cfg['dt_info']['stu_count']
        self.num_q = self.datafmt_cfg['dt_info']['exer_count']
        self.num_c = self.datafmt_cfg['dt_info']['cpt_count']
        self.emb_size = self.model_cfg['emb_size']
        self.dropout = self.model_cfg['dropout']
        self.difficult_levels = self.model_cfg['difficult_levels']
        self.num_steps = self.model_cfg['num_steps']
        
    def build_model(self):
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.dropout = Dropout(self.dropout)
        self.knowledge = Variable(torch.randn(1, self.emb_size), requires_grad=True)

        self.q_emb = Embedding(self.num_q , self.emb_size, device=self.device, padding_idx=0)
        self.c_emb = Embedding(self.num_c , self.emb_size, device=self.device, padding_idx=0)
        self.sd_emb = Embedding(self.difficult_levels , self.emb_size, device=self.device, padding_idx=0)
        self.qd_emb = Embedding(self.difficult_levels , self.emb_size, device=self.device, padding_idx=0)
        self.a_emb = Embedding(2, self.emb_size, device=self.device)

        self.linear_1 = Linear(4 * self.emb_size, self.emb_size)
        self.linear_2 = Linear(1 * self.emb_size, self.emb_size)
        self.linear_3 = Linear(1 * self.emb_size, self.emb_size)
        self.linear_4 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_5 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_6 = Linear(4 * self.emb_size, self.emb_size)



    def forward(self, exer_seq, label_seq,  **kwargs):

        self.bs = len(exer_seq)
        q = exer_seq[:,:-1]
        c = kwargs['cpt_seq'][:,:-1]
        sd = kwargs['cd_seq'][:,:-1].int()
        qd = kwargs['qd_seq'][:,:-1].int()
        a = label_seq[:,:-1].int()
        q_emb = self.q_emb(Variable(q))
        c_emb = self.c_emb(Variable(c))
        sd_emb = self.sd_emb(Variable(sd))
        qd_emb = self.qd_emb(Variable(qd))
        a_emb = self.a_emb(Variable(a))

        qshft = exer_seq[:, 1:]
        cshft = kwargs['cpt_seq'][:, 1:]
        sdshft = kwargs['cd_seq'][:, 1:].int()
        qdshft = kwargs['qd_seq'][:, 1:].int()
        target_q = self.q_emb(Variable(qshft))
        target_c = self.c_emb(Variable(cshft))
        target_sd = self.sd_emb(Variable(sdshft))
        target_qd = self.qd_emb(Variable(qdshft))

        input_data = torch.cat((q_emb, c_emb, sd_emb, qd_emb), -1)
        input_data = self.linear_1(input_data)

        target_data = torch.cat((target_q, target_c, target_sd, target_qd), -1)
        target_data = self.linear_1(target_data)

        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        sd_emb = torch.cat((padd, sd_emb), 1)
        slice_sd_embedding = sd_emb.split(1, dim=1)

        shape = list(a_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        a_emb = torch.cat((padd, a_emb), 1)
        slice_a_embedding = a_emb.split(1, dim=1)

        shape = list(input_data.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        input_data = torch.cat((padd, input_data), 1)
        slice_input_data = input_data.split(1, dim=1)

        qd_emb = torch.cat((padd, qd_emb), 1)
        slice_qd_embedding = qd_emb.split(1, dim=1)

        k = self.knowledge.repeat(self.bs, 1).to(self.device)

        h = list()
        for i in range(1, self.num_steps + 1):
            sd_1 = torch.squeeze(slice_sd_embedding[i], 1)
            a_1 = torch.squeeze(slice_a_embedding[i], 1)
            qd_1 = torch.squeeze(slice_qd_embedding[i], 1)
            input_data_1 = torch.squeeze(slice_input_data[i], 1)

            qq = k - input_data_1

            gates_SDF = self.linear_2(qq)
            gates_SDF = self.sigmoid(gates_SDF)
            SDFt = self.linear_3(qq)
            SDFt = self.tanh(SDFt)
            SDFt = self.dropout(SDFt)

            SDFt = gates_SDF * SDFt

            x = torch.cat((SDFt, a_1), -1)
            gates_PKA = self.linear_4(x)
            gates_PKA = self.sigmoid(gates_PKA)

            PKAt = self.linear_5(x)
            PKAt = self.tanh(PKAt)

            PKAt = gates_PKA * PKAt

            ins = torch.cat((k, a_1, sd_1, qd_1), -1)
            gates_KSU = self.linear_6(ins)
            gates_KSU = self.sigmoid(gates_KSU)

            k = gates_KSU * k + (1 - gates_KSU) * PKAt

            h_i = torch.unsqueeze(k, dim=1)
            h.append(h_i)

        output = torch.cat(h, axis=1)
        logits = torch.sum(target_data * output, dim=-1)
        y = self.sigmoid(logits)

        return y

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        # y_pd = y_pd[:, :-1]
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
        y_pd= self(**kwargs)
        # y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss,
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
