from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class SAKT(GDBaseModel):
    default_cfg = {
        'emb_size': 128,
        'n_attn_heads': 8,
        'max_length': 100,
        'dropout_rate': 0.2,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']

    def build_model(self):
        self.M = nn.Embedding(self.n_item * 2, self.modeltpl_cfg['emb_size'])
        self.E = nn.Embedding(self.n_item, self.modeltpl_cfg['emb_size'])
        self.P = nn.Parameter(torch.Tensor(self.modeltpl_cfg['max_length'] - 1, self.modeltpl_cfg['emb_size']))

        self.attn = nn.MultiheadAttention(
            self.modeltpl_cfg['emb_size'], self.modeltpl_cfg['n_attn_heads'], dropout=self.modeltpl_cfg['dropout_rate']
        )
        self.attn_dropout = nn.Dropout(self.modeltpl_cfg['dropout_rate'])
        self.attn_layer_norm = nn.LayerNorm(self.modeltpl_cfg['emb_size'])

        self.FFN = nn.Sequential(
            nn.Linear(self.modeltpl_cfg['emb_size'], self.modeltpl_cfg['emb_size']),
            nn.ReLU(),
            nn.Dropout(self.modeltpl_cfg['dropout_rate']),
            nn.Linear(self.modeltpl_cfg['emb_size'], self.modeltpl_cfg['emb_size']),
            nn.Dropout(self.modeltpl_cfg['dropout_rate']),
        )
        self.FFN_layer_norm = nn.LayerNorm(self.modeltpl_cfg['emb_size'])

        self.pred = nn.Linear(self.modeltpl_cfg['emb_size'], 1)

    def forward(self, exer_seq, label_seq, **kwargs):
        q, r, qry = exer_seq[:, :-1].to(torch.int64), label_seq[:, :-1].to(torch.int64), exer_seq[:, 1:].to(torch.int64)

        x = q + self.n_item* r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)
        M = M + P

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool().to(self.device)

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        

        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights
    
    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd, _ = self(**kwargs)
        # y_pd = y_pd[:, :-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
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
        y_pd, _ = self(**kwargs)  # use forward
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


        