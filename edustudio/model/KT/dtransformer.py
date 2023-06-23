from ..gd_basemodel import GDBaseModel
import random
import math
import torch.nn as nn
import torch
import torch.nn.functional as F

MIN_SEQ_LEN = 5

class DTransformer(GDBaseModel):
    default_cfg = {
        'n_knowledges': 16,
        'hidden_size': 128,
        'dim_fc': 256,
        'num_layers': 3,
        'num_heads': 8,
        'lambda_cl': 0.1,
        'prediction_window': 1,
        'dropout': 0.2,
        'projection_alhead_cl': 0,
        'cl_loss': True,
        'hard_negative': True,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_exer = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def build_model(self):
        self.cpt_embed = nn.Embedding(self.n_cpt + 1, self.modeltpl_cfg['hidden_size'])
        self.label_embed = nn.Embedding(2, self.modeltpl_cfg['hidden_size'])
        
        self.cpt_diff_embed = nn.Embedding(self.n_cpt + 1, self.modeltpl_cfg['hidden_size'])
        self.label_diff_embed = nn.Embedding(2, self.modeltpl_cfg['hidden_size'])
        self.exer_diff_embed = nn.Embedding(self.n_exer + 1, 1)  # difficult parameter

        self.block1 = DTransformerLayer(self.modeltpl_cfg['hidden_size'], self.modeltpl_cfg['num_heads'], self.modeltpl_cfg['dropout'])
        self.block2 = DTransformerLayer(self.modeltpl_cfg['hidden_size'], self.modeltpl_cfg['num_heads'], self.modeltpl_cfg['dropout'], kq_same=False)

        self.know_params = nn.Parameter(torch.empty(self.modeltpl_cfg['n_knowledges'], self.modeltpl_cfg['hidden_size']))
        nn.init.uniform_(self.know_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(self.modeltpl_cfg['hidden_size'] * 2, self.modeltpl_cfg['dim_fc']),
            nn.GELU(),
            nn.Dropout(self.modeltpl_cfg['dropout']),
            nn.Linear(self.modeltpl_cfg['dim_fc'], self.modeltpl_cfg['dim_fc'] // 2),
            nn.GELU(),
            nn.Dropout(self.modeltpl_cfg['dropout']),
            nn.Linear(self.modeltpl_cfg['dim_fc'] // 2, 1),
        )

        if self.modeltpl_cfg['projection_alhead_cl']:
            self.proj = nn.Sequential(nn.Linear(self.modeltpl_cfg['hidden_size'], self.modeltpl_cfg['hidden_size']), nn.GELU())
        else:
            self.proj = None
        
    def forward(self, exer_seq, cpt_unfold_seq, label_seq, mask_seq, is_train=True, n=1,**kwargs):
        lens = (mask_seq > 0).sum(dim=1)
        q_emb, a_emb, exer_diff = self.embedding(cpt_unfold_seq, label_seq, exer_seq)
        
        if self.modeltpl_cfg['num_layers'] == 1:
            hq = q_emb.clone()
            ha = a_emb.clone()
            m, e_score = self.block1(hq, hq, ha, lens, peek_cur=True)
        elif self.modeltpl_cfg['num_layers'] == 2:
            hq = q_emb.clone()
            ha, _ = self.block1(a_emb, a_emb, a_emb, lens, peek_cur=True)
            m, e_score = self.block1(hq, hq, ha, lens, peek_cur=True)
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            ha, _ = self.block1(a_emb, a_emb, a_emb, lens, peek_cur=True)
            m, e_score = self.block1(hq, hq, ha, lens, peek_cur=True)

        bs, seqlen = m.size(0), m.size(1)

        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * self.modeltpl_cfg['n_knowledges'], seqlen, self.modeltpl_cfg['hidden_size'])
        )
        hq = hq.unsqueeze(1).expand(-1, self.modeltpl_cfg['n_knowledges'], -1, -1).reshape_as(query)  # key embedding
        m = m.unsqueeze(1).expand(-1, self.modeltpl_cfg['n_knowledges'], -1, -1).reshape_as(query)  # value embedding
        z, c_score = self.block2(query, hq, m, torch.repeat_interleave(lens, self.modeltpl_cfg['n_knowledges']), peek_cur=False)
        z = (
            z.view(bs, self.modeltpl_cfg['n_knowledges'], seqlen, self.modeltpl_cfg['hidden_size'])  # unpack dimensions
            .transpose(1, 2)  # (bs, seqlen, n_know, d_model)
            .contiguous()
            .view(bs, seqlen, -1)
        )
        c_score = (
            c_score.view(bs, self.modeltpl_cfg['n_knowledges'], self.modeltpl_cfg['num_heads'], seqlen, seqlen)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        if not is_train:
            return z, q_emb
        else:
            query = q_emb[:, n - 1 :, :]
            h = self.readout(z[:, : query.size(1), :], query)
            y = torch.sigmoid(self.out(torch.cat([query, h], dim=-1))).squeeze(-1)

            return y, z, q_emb, (exer_diff**2).mean() * 1e-3, (e_score, c_score)
        
    def readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.modeltpl_cfg['n_knowledges'], -1)
        )
        value = z.reshape(bs * seqlen, self.modeltpl_cfg['n_knowledges'], -1)

        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.modeltpl_cfg['n_knowledges'])
        alpha = torch.softmax(beta, -1)

        return torch.matmul(alpha, value).view(bs, seqlen, -1)
    
    def embedding(self, cpt_seq, label_seq, exer_seq=None):
        cpt_emb = self.cpt_embed(cpt_seq)
        label_emb = self.label_embed(label_seq.long()) + cpt_emb

        exer_diff = 0.0
        exer_diff = self.exer_diff_embed(exer_seq)

        cpt_diff_emb = self.cpt_diff_embed(cpt_seq)  # Concept variations
        cpt_emb += exer_diff * cpt_diff_emb

        label_diff_emb = self.label_diff_embed(label_seq.long()) + cpt_diff_emb
        label_emb += exer_diff * label_diff_emb

        return cpt_emb, label_emb, exer_diff

    @torch.no_grad()
    def predict(self, n=1, **kwargs):  
        z, q_emb, = self(**kwargs, is_train=False)
        query = q_emb[:, n-1 :, :]
        h = self.readout(z[:, : query.size(1), :], query)
        y_pd = torch.sigmoid(self.out(torch.cat([query, h], dim=-1))).squeeze(-1)

        y_pd = y_pd if n == 1 else y_pd[:, :-n+1]
        y_pd = y_pd[kwargs['mask_seq'][:, n-1:] == 1]

        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, n-1:]
            y_gt = y_gt[kwargs['mask_seq'][:, n-1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def get_loss(self, **kwargs):
        y_pd, _, _, reg_loss, _ = self(**kwargs, is_train=True)
        y_pd = y_pd[kwargs['mask_seq'] == 1]

        y_gt = kwargs['label_seq']
        y_gt = y_gt[kwargs['mask_seq'] == 1]

        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt, reduction="mean"
        ) + reg_loss
        return {
            'loss_main': loss
        }

    def get_cl_loss(self, **kwargs):   
        bs = kwargs['exer_seq'].size(0)
        lens = (kwargs['mask_seq'] > 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(**kwargs)
        
        # Data Augmentation
        cpt_seq_ = kwargs['cpt_unfold_seq'].clone()
        label_seq_ = kwargs['label_seq'].clone()
        exer_seq_ = kwargs['exer_seq'].clone()
        # Manipulate order: Swap Adjacent Items
        for b in range(bs):
            idx = random.sample(range(lens[b] - 1), max(1, int(lens[b] * self.modeltpl_cfg['dropout'])))
            for i in idx:
                cpt_seq_[b, i], cpt_seq_[b, i + 1] = cpt_seq_[b, i + 1], cpt_seq_[b, i]
                label_seq_[b, i], label_seq_[b, i + 1] = label_seq_[b, i + 1], label_seq_[b, i]
                exer_seq_[b, i], exer_seq_[b, i + 1] = exer_seq_[b, i + 1], exer_seq_[b, i]
        # Hard negative 
        label_seq_flip = kwargs['label_seq'].clone() if self.modeltpl_cfg['hard_negative'] else label_seq_
        # Manipulate score: Flip Response
        for b in range(bs):
            idx = random.sample(range(lens[b] - 1), max(1, int(lens[b] * self.modeltpl_cfg['dropout'])))
            for i in idx:
                label_seq_flip[b, i] = 1 - label_seq_flip[b, i]
        if not self.modeltpl_cfg['hard_negative']: label_seq_ = label_seq_flip

        # Model loss
        logits, z_1, q_emb, reg_loss, _ = self(**kwargs, is_train=True)
        logits_masked = logits[kwargs['mask_seq'] == 1]

        _, z_2, *_ = self(exer_seq_, cpt_seq_, label_seq_, kwargs['mask_seq'], is_train=True)
        if self.modeltpl_cfg['hard_negative']:
            _, z_3, *_ = self(kwargs['exer_seq'], kwargs['cpt_unfold_seq'], label_seq_flip, kwargs['mask_seq'], is_train=True)

        # CL loss
        input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.modeltpl_cfg['hard_negative']:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input = torch.cat([input, hard_neg], dim=1)
        target = (
            torch.arange(kwargs['label_seq'].size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )
        cl_loss = F.cross_entropy(
            input=input, target=target
        )

        # Prediction loss
        labels_masked = kwargs['label_seq'][kwargs['mask_seq'] == 1]
        pred_loss = F.binary_cross_entropy(
            input=logits_masked, target=labels_masked, reduction="mean"
        )
        
        for i in range(1, self.modeltpl_cfg["prediction_window"]):
            y_gt = kwargs['label_seq'][:, i:]
            y_gt = y_gt[kwargs['mask_seq'][:, i:] == 1]

            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y_pt = torch.sigmoid(self.out(torch.cat([query, h], dim=-1))).squeeze(-1)
            y_pd = y_pd[kwargs['mask_seq'][:, i:] == 1]

            pred_loss += F.binary_cross_entropy(
            input=y_pt, target=y_gt, reduction="mean"
        )
        pred_loss /= self.modeltpl_cfg["prediction_window"]

        return {
            'loss_main': pred_loss + cl_loss * self.modeltpl_cfg['lambda_cl'] + reg_loss,
        }

    def get_loss_dict(self, **kwargs):
        if self.modeltpl_cfg['cl_loss']:
            return self.get_cl_loss(**kwargs)
        else:
            return self.get_loss(**kwargs)

    def sim(self, z1, z2):
        bs, seqlen, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.modeltpl_cfg['n_knowledges'], -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.modeltpl_cfg['n_knowledges'], -1)
        if self.modeltpl_cfg['projection_alhead_cl']:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05

class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device
    
    def forward(self, query, key, values, lens, peek_cur=False):
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)  
        mask = mask.bool()[None, None, :, :].to(self.device())

        # mask manipulation: Drop Item
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous() 

            for b in range(query.size(0)):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0
        
        # apply transformer layer
        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur
        )
        query = query + self.dropout(query_)

        return self.layer_norm(query), scores

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
    
    def forward(self, q, k, v, mask, maxout=False):
        bs = q.size(0)
        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)

        # calculate attention using function we will define next
        v_, scores = attention(q, k, v, mask, self.gammas, maxout)

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, scores

def attention(q, k, v, mask, gamma=None, maxout=False):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # include temporal effect
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]  
            dist_scores = torch.clamp(  
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect
    
    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage

    # max-out scores (bs, n_heads, seqlen, seqlen)
    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale
    
    # calculate output
    output = torch.matmul(scores, v)
    return output, scores

