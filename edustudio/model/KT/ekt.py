from ..gd_basemodel import GDBaseModel
from .eernn import EERNNM, EERNNA
import torch.nn as nn
import torch
import torch.nn.functional as F


class EKTM(EERNNM):
    default_cfg = {
        'd_k': 25,
    }

    def build_cfg(self):
        super().build_cfg()
        self.n_cpt = self.datafmt_cfg['dt_info']['cpt_count']

    def build_model(self):
        super().build_model()
        self.cpt_mem_emb = nn.Embedding(self.n_cpt, self.model_cfg['d_k'])
        self.W_K = nn.Embedding(self.n_cpt, self.model_cfg['d_k'])

    def forward(self, exer_seq, label_seq, cpt_seq, cpt_seq_mask, **kwargs):
        exers, idx = exer_seq.unique(return_inverse=True)
        exer_emb = self.get_exer_embedding(exers)[idx]
        fused_exer_emb = exer_emb.repeat(1,1,2)
        cond = torch.where(label_seq == 1)
        fused_exer_emb[cond[0], cond[1], self.d_v*2:] = 0.0
        cond = torch.where(label_seq == 0)
        fused_exer_emb[cond[0], cond[1], 0:self.d_v*2] = 0.0

        # compute beta
        V_T = (self.W_K(cpt_seq) * cpt_seq_mask.unsqueeze(dim=-1)).sum(dim=-2)
        Beta = torch.matmul(V_T, self.cpt_mem_emb.weight.T).softmax(dim=-1)
        
        X = (Beta.unsqueeze(dim=-1) * fused_exer_emb.unsqueeze(dim=-2))
        h = None
        for k in range(self.n_cpt):
            h_k, _ = self.stu_inter_seq_model(X[:,:,k,:].squeeze())
            # H_list.append((h_k * Beta[:,:,k].unsqueeze(dim=-1)).unsqueeze(dim=-2))
            if h is not None:
                h += (h_k * Beta[:,:,k].unsqueeze(dim=-1))
            else:
                h = (h_k * Beta[:,:,k].unsqueeze(dim=-1))
            
        # h = torch.cat(H_list, dim=-2).sum(dim=-2)
        feat = torch.cat([h[:,0:-1,:], exer_emb[:,1:,:]], dim=-1)
        return self.pd_layer(feat)


class EKTA(EERNNA):
    pass

