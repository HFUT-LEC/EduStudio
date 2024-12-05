r"""
EKT
##########################################

Reference:
    Qi Liu et al. "EKT: Exercise-Aware Knowledge Tracing for Student Performance Prediction" in TKDE 2019.

Reference Code:
    https://github.com/bigdata-ustc/ekt

"""

from .eernn import EERNNM, EERNNA
import torch.nn as nn
import torch
import torch.nn.functional as F


class KnowledgeModel(nn.Module):
    """
    Transform Knowledge index to knowledge embedding
    """

    def __init__(self, know_len, know_emb_size):
        super(KnowledgeModel, self).__init__()
        self.knowledge_embedding = nn.Embedding(know_len+1, know_emb_size, padding_idx=0)

    def forward(self, cpt_seq, cpt_seq_mask):
        t = cpt_seq + 1
        t[cpt_seq_mask == 0] = 0
        return self.knowledge_embedding(t).sum(dim=1)


class EKTM(EERNNM):
    default_cfg = {
        'd_k': 25,
    }

    def build_cfg(self):
        super().build_cfg()
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.d_k = self.modeltpl_cfg['d_k']

    def build_model(self):
        super().build_model()
        self.cpt_mem_emb = nn.Embedding(self.n_cpt, self.d_k)
        self.W_K = nn.Embedding(self.n_cpt, self.d_k)
        self.h_initial = nn.Parameter(torch.zeros(self.n_cpt, self.d_h))
        self.knowledge_model = KnowledgeModel(self.n_cpt, self.d_k)
        self.stu_inter_seq_model = nn.GRU(4 * self.d_v, self.d_h)

    def _init_params(self):
        super()._init_params()
        self.h_initial.data.uniform_(-1, 1)

    def forward(self, exer_seq, label_seq, cpt_seq, cpt_seq_mask, **kwargs):
        exers, idx = exer_seq.unique(return_inverse=True)
        exer_emb = self.get_exer_embedding(exers)[idx]
        fused_exer_emb = exer_emb.repeat(1,1,2)
        cond = torch.where(label_seq == 1)
        fused_exer_emb[cond[0], cond[1], self.d_v*2:] = 0.0
        cond = torch.where(label_seq == 0)
        fused_exer_emb[cond[0], cond[1], 0:self.d_v*2] = 0.0


        seq_len = exer_seq.shape[1]
        h = self.h_initial
        pd_list = []
        for t in range(seq_len-1): # iter each time
            v_t = self.knowledge_model(cpt_seq[:,t,:], cpt_seq_mask[:,t,:])
            beta = torch.mm(self.cpt_mem_emb.weight, v_t.T).T.softmax(dim=1)

            # 计算预测值
            s = torch.mm(beta, h) # B*K, K*d -> B*d
            pred_v = torch.cat([exer_emb[:, t+1, :], s], dim=-1) # 1*B, 即每个学生在此刻对下一题的预测
            pd_t = self.pd_layer(pred_v)

            # 得到hidden vector
            xk = beta.unsqueeze(dim=-1) * fused_exer_emb[:,[t],:]
            _, h = self.stu_inter_seq_model(xk, h.unsqueeze(0))
            h = h.squeeze(0)
            pd_list.append(pd_t)
        return torch.concat(pd_list, dim=-1)


class EKTA(EKTM):
    def forward(self, exer_seq, label_seq, cpt_seq, cpt_seq_mask, **kwargs):
        exers, idx = exer_seq.unique(return_inverse=True)
        exer_emb = self.get_exer_embedding(exers)[idx]
        fused_exer_emb = exer_emb.repeat(1,1,2)
        cond = torch.where(label_seq == 1)
        fused_exer_emb[cond[0], cond[1], self.d_v*2:] = 0.0
        cond = torch.where(label_seq == 0)
        fused_exer_emb[cond[0], cond[1], 0:self.d_v*2] = 0.0

        seq_len = exer_seq.shape[1]
        h = self.h_initial
        pd_list = []
        h_list = [h]
        for t in range(seq_len-1): # iter each time
            v_t = self.knowledge_model(cpt_seq[:,t,:], cpt_seq_mask[:,t,:])
            beta = torch.mm(self.cpt_mem_emb.weight, v_t.T).T.softmax(dim=1)

            # 计算预测值
            cos_sim = F.cosine_similarity(exer_emb[:, [t+1], :], exer_emb[:, 0:t+1, :], dim=-1)
            h_new = (torch.stack(h_list, dim=0).unsqueeze(0).repeat(exer_seq.shape[0], 1, 1,1) *  cos_sim.unsqueeze(dim=-1).unsqueeze(dim=-1)).sum(dim=1)

            s = (beta.unsqueeze(-1)* h_new).sum(dim=1) # B*K, K*d -> B*d
            pred_v = torch.cat([exer_emb[:, t+1, :], s], dim=-1) # 1*B, 即每个学生在此刻对下一题的预测
            pd_t = self.pd_layer(pred_v)

            # 得到hidden vector
            xk = beta.unsqueeze(dim=-1) * fused_exer_emb[:,[t],:]
            _, h = self.stu_inter_seq_model(xk, h.unsqueeze(0))
            h = h.squeeze(0)
            h_list.append(h)
            pd_list.append(pd_t)

        return torch.concat(pd_list, dim=-1)
