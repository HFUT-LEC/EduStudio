r"""
EERNN
##########################################

Reference:
    Yu Su et al. "Exercise-Enhanced Sequential Modeling for Student Performance Prediction" in AAAI 2018.

Reference Code:
    https://github.com/shaoliangliang1996/EERNN

"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class EERNNM(GDBaseModel):
    default_cfg = {
        'd_v': 100,
        'd_h': 100,
    }
    
    def add_extra_data(self, **kwargs):
        self.w2v_text_emb = kwargs['w2v_word_emb']
        self.exer_content = kwargs['exer_content'].to(self.device)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_word = self.datatpl_cfg['dt_info']['word_count']
        self.d_0 = self.datatpl_cfg['dt_info']['word_emb_dim']
        self.d_v = self.modeltpl_cfg['d_v']
        self.d_h = self.modeltpl_cfg['d_h']

    def build_model(self):
        self.word_emb = nn.Embedding(self.n_word, self.d_0, padding_idx=0)
        self.exer_text_seq_model = nn.LSTM(self.d_0, self.d_v, batch_first=True, bidirectional=True)
        self.stu_inter_seq_model = nn.LSTM(4 * self.d_v, self.d_h, batch_first=True)
        self.pd_layer = nn.Sequential(
            nn.Linear(in_features=self.d_h + 2*self.d_v, out_features=(self.d_h + 2*self.d_v) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=(self.d_h + 2*self.d_v) // 2, out_features=1),
            nn.Sigmoid()
        )

    def _init_params(self):
        super()._init_params()
        self.word_emb.weight.data.copy_(torch.from_numpy(self.w2v_text_emb))

    def get_exer_embedding(self, exers=None):
        if exers is not None:
            t = self.word_emb(self.exer_content[exers])
        else:
            t = self.word_emb(self.exer_content)
        mix_h, _ = self.exer_text_seq_model(t)
        forward_h, reverse_h = torch.chunk(mix_h, 2, dim=-1)
        v_m = torch.cat([forward_h, reverse_h.flip(dims=[0])], dim=-1)
        return torch.max(v_m, dim=1)[0]

    def forward(self, exer_seq, label_seq, **kwargs):
        exers, idx = exer_seq.unique(return_inverse=True)
        exer_emb = self.get_exer_embedding(exers)[idx]
        fused_exer_emb = exer_emb.repeat(1,1,2)
        cond = torch.where(label_seq == 1)
        fused_exer_emb[cond[0], cond[1], self.d_v*2:] = 0.0
        cond = torch.where(label_seq == 0)
        fused_exer_emb[cond[0], cond[1], 0:self.d_v*2] = 0.0

        h, _ = self.stu_inter_seq_model(fused_exer_emb)
        feat = torch.cat([h[:,0:-1,:], exer_emb[:,1:,:]], dim=-1)
        return self.pd_layer(feat)

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs).squeeze(dim=-1)
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
        y_pd = self(**kwargs).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)


class EERNNA(EERNNM):
    def forward(self, exer_seq, label_seq, **kwargs):
        exers, idx = exer_seq.unique(return_inverse=True)
        exer_emb_unique = self.get_exer_embedding(exers)
        exer_emb = exer_emb_unique[idx]
        fused_exer_emb = exer_emb.repeat(1,1,2)
        cond = torch.where(label_seq == 1)
        fused_exer_emb[cond[0], cond[1], self.d_v*2:] = 0.0
        cond = torch.where(label_seq == 0)
        fused_exer_emb[cond[0], cond[1], 0:self.d_v*2] = 0.0

        h, _ = self.stu_inter_seq_model(fused_exer_emb)
        
        # 加权h

        # a_norm = exer_emb_unique / exer_emb_unique.norm(dim=1)[:, None]
        # cos_sim_mat = torch.mm(a_norm, a_norm.transpose(0,1)) # cos similarity
        h_new = []
        for tid in range(exer_seq.shape[-1] - 1):
            cos_sim = F.cosine_similarity(exer_emb[:, [tid+1], :], exer_emb[:, 0:tid+1, :], dim=-1)
            h_new.append((h[:,0:tid+1, :] *  cos_sim.unsqueeze(dim=-1)).sum(dim=1, keepdim=True))

        h_new = torch.cat(h_new, dim=1)
        feat = torch.cat([h_new, exer_emb[:,1:,:]], dim=-1)
        return self.pd_layer(feat)
