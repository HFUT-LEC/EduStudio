r"""
CL4KT
################################################
Reference:

    Wongsung Lee et al. "Contrastive Learning for Knowledge Tracing." in WWW2022.

Reference Code:

    https://github.com/UpstageAI/cl4kt
"""
from ..gd_basemodel import GDBaseModel
import random
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from edustudio.utils.common import set_same_seeds
import numpy as np
from ...datatpl.utils import PadSeqUtil


class CL4KT(GDBaseModel):
    """
    CL4KT introduces a contrastive learning framework for knowledge tracing that learns effective representations by 
    pulling similar learning histories together and pushing dissimilar learning histories apart in representation space.
    """
    default_cfg = {
        'emb_size': 64,
        'hidden_size': 64,
        'num_blocks': 2,
        'num_attn_heads': 8,
        'kq_same': True,
        'dim_fc': 512,
        'd_ff': 1024,
        'l2': 0.0,
        'dropout': 0.2,
        'reg_cl': 0.1,
        'mask_prob': 0.2,
        'crop_prob': 0.3,
        'permute_prob': 0.3,
        'replace_prob': 0.3,
        'hard_negative': True,
        'temp': 0.05,
        'hard_negative_weight': 1.0,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        """
        Get exercise nums and concepts nums.
        """
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def build_model(self):
        """
        Initial model operator.
        """
        self.question_embed = nn.Embedding(
            self.n_cpt  + 2, self.modeltpl_cfg['emb_size']
        )
        self.interaction_embed = nn.Embedding(
            2 * (self.n_cpt + 2), self.modeltpl_cfg['emb_size']
        )
        self.sim = Similarity(temp=self.modeltpl_cfg["temp"])
        self.cpt_encoder = nn.ModuleList(
            [
                CL4KTTransformerLayer(
                    self.modeltpl_cfg['hidden_size'],
                    self.modeltpl_cfg['hidden_size'] // self.modeltpl_cfg['num_attn_heads'],
                    self.modeltpl_cfg['d_ff'],
                    self.modeltpl_cfg['num_attn_heads'],
                    self.modeltpl_cfg['dropout'],
                    self.modeltpl_cfg['kq_same']
                    )
                for _ in range(self.modeltpl_cfg['num_blocks'])
            ]
        )
        self.interaction_encoder = nn.ModuleList(
            [
                CL4KTTransformerLayer(
                    self.modeltpl_cfg['hidden_size'],
                    self.modeltpl_cfg['hidden_size'] // self.modeltpl_cfg['num_attn_heads'],
                    self.modeltpl_cfg['d_ff'],
                    self.modeltpl_cfg['num_attn_heads'],
                    self.modeltpl_cfg['dropout'],
                    self.modeltpl_cfg['kq_same']
                    )
                for _ in range(self.modeltpl_cfg['num_blocks'])
            ]
        )
        self.knoweldge_retriever = nn.ModuleList(
            [
                CL4KTTransformerLayer(
                    self.modeltpl_cfg['hidden_size'],
                    self.modeltpl_cfg['hidden_size'] // self.modeltpl_cfg['num_attn_heads'],
                    self.modeltpl_cfg['d_ff'],
                    self.modeltpl_cfg['num_attn_heads'],
                    self.modeltpl_cfg['dropout'],
                    self.modeltpl_cfg['kq_same']
                    )
                for _ in range(self.modeltpl_cfg['num_blocks'])
            ]
        )

        self.out = nn.Sequential(
            nn.Linear(self.modeltpl_cfg['hidden_size'] * 2, self.modeltpl_cfg['dim_fc']),
            nn.GELU(),
            nn.Dropout(self.modeltpl_cfg['dropout']),
            nn.Linear(self.modeltpl_cfg['dim_fc'], self.modeltpl_cfg['dim_fc'] // 2),
            nn.GELU(),
            nn.Dropout(self.modeltpl_cfg['dropout']),
            nn.Linear(self.modeltpl_cfg['dim_fc'] // 2, 1),
        )

    def forward(self, **kwargs):
        cpt_embed = self.question_embed(kwargs['cpt_unfold_seq'])
        inter_embed = self.get_interaction_embed(kwargs['cpt_unfold_seq'], kwargs['label_seq'].long(), kwargs['mask_seq'])

        x, y = cpt_embed, inter_embed
        for block in self.cpt_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True) 
        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=True)
        for block in self.knoweldge_retriever:
            x, att = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        
        retrieved_knowledge = torch.cat([x, cpt_embed], dim=-1)
        # y_pd = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()
        y_pd = torch.sigmoid(self.out(retrieved_knowledge))

        return y_pd, att
    
    def get_interaction_embed(self, cpts, responses, attention_mask):
        """
        Get interaction embedding by concepts and responses.
        """
        # masked_responses = responses * (attention_mask > 0).long()
        interactions = cpts + self.n_cpt * responses

        return self.interaction_embed(interactions)

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd,_ = self(**kwargs)
        y_pd = y_pd[:, 1:]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1].squeeze(dim=1)
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def get_main_loss(self, **kwargs):
        """Loss calculation entry, including response prediciton loss and contrastive loss.

        Parameters
        ----------

            kwargs: dict
                kwargs is input feature embedding tensor including stu_id、exer_seq、cpt_unfold_seq、label_seq and mask_seq.
            stu_id: list
                Student id information.
            exer_seq: list
                Exercise is done by students.
            cpt_unfold_seq: list
                Concepts is done by students.
            label_seq: list
                It reprsents whether the exercise is done correctly.
            mask_seq: list
                Real data location.

        Returns
        -------
        torch.FloatTensor

            Response Prediction Loss and Contrastive Loss.
        """
        _, aug_c_seq_1, aug_r_seq_1, negative_r_seq, attention_mask_1 = self.augment_kt_seqs(**kwargs)
        _, aug_c_seq_2, aug_r_seq_2, _, attention_mask_2 = self.augment_kt_seqs(seed_change=True,**kwargs)
        aug_r_seq_1, aug_r_seq_2, negative_r_seq = aug_r_seq_1.long(), aug_r_seq_2.long(), negative_r_seq.long()
        attention_mask_1, attention_mask_2 = attention_mask_1.long(), attention_mask_2.long()

        # CL loss
        cpt_i_embed, cpt_j_embed = self.question_embed(aug_c_seq_1), self.question_embed(aug_c_seq_2)
        inter_i_embed, inter_j_embed = self.get_interaction_embed(aug_c_seq_1, aug_r_seq_1, attention_mask_1), self.get_interaction_embed(aug_c_seq_2, aug_r_seq_2, attention_mask_2)
        if self.modeltpl_cfg['hard_negative']: inter_k_embed = self.get_interaction_embed(kwargs['cpt_unfold_seq'], negative_r_seq, kwargs['mask_seq'])

        cpt_i_score, cpt_j_score = cpt_i_embed, cpt_j_embed
        inter_i_score, inter_j_score = inter_i_embed, inter_j_embed
        for block in self.cpt_encoder:
            cpt_i_score, _ = block(mask=2, query=cpt_i_score, key=cpt_i_score, values=cpt_i_embed, apply_pos=False)
            cpt_j_score, _ = block(mask=2, query=cpt_j_score, key=cpt_j_score, values=cpt_j_embed, apply_pos=False)
        for block in self.interaction_encoder:
            inter_i_score, _ = block(mask=2, query=inter_i_score, key=inter_i_score, values=inter_i_embed, apply_pos=False)
            inter_j_score, _ = block(mask=2, query=inter_j_score, key=inter_j_score, values=inter_j_embed, apply_pos=False)
            if self.modeltpl_cfg['hard_negative']: inter_k_embed, _ = block(mask=2, query=inter_k_embed, key=inter_k_embed, values=inter_k_embed, apply_pos=False)
        
        pool_cpt_i_score = (cpt_i_score * attention_mask_1.unsqueeze(-1)).sum(1) / attention_mask_1.sum(-1).unsqueeze(-1)
        pool_cpt_j_score = (cpt_j_score * attention_mask_2.unsqueeze(-1)).sum(1) / attention_mask_2.sum(-1).unsqueeze(-1)
        cpt_cos_sim = self.sim(pool_cpt_i_score.unsqueeze(1), pool_cpt_j_score.unsqueeze(0))
        cpt_labels = torch.arange(cpt_cos_sim.size(0)).long().to(aug_c_seq_1.device)
        cpt_cls_loss = F.cross_entropy(
            input=cpt_cos_sim, target=cpt_labels, reduction="mean"
        )

        pool_inter_i_score = (inter_i_score * attention_mask_1.unsqueeze(-1)).sum(1) / attention_mask_1.sum(-1).unsqueeze(-1)
        pool_inter_j_score = (inter_j_score * attention_mask_2.unsqueeze(-1)).sum(1) / attention_mask_2.sum(-1).unsqueeze(-1)
        inter_cos_sim = self.sim(pool_inter_i_score.unsqueeze(1), pool_inter_j_score.unsqueeze(0))
        if self.modeltpl_cfg['hard_negative']:
            pool_inter_k_score = (inter_k_embed * kwargs['mask_seq'].unsqueeze(-1)).sum(1) / kwargs['mask_seq'].sum(-1).unsqueeze(-1)
            neg_inter_cos_sim = self.sim(pool_inter_i_score.unsqueeze(1), pool_inter_k_score.unsqueeze(0))
            inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)
        inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(aug_c_seq_1.device)
        if self.modeltpl_cfg['hard_negative']:
            weights = torch.tensor(
                    [
                        [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                        + [0.0] * i
                        + [self.modeltpl_cfg['hard_negative_weight']]
                        + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                        for i in range(neg_inter_cos_sim.size(-1))
                    ]
            ).to(aug_c_seq_1.device)
            inter_cos_sim = inter_cos_sim + weights
        inter_cls_loss = F.cross_entropy(
            input=inter_cos_sim, target=inter_labels, reduction="mean"
        )
        cl_loss = cpt_cls_loss + inter_cls_loss

        # Predict
        y_pd, _ = self(**kwargs)

        y_pd = y_pd[:, 1:]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1].squeeze(dim=1)
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt, reduction="mean"
        )
        return {
            'loss_main': loss + cl_loss * self.modeltpl_cfg['reg_cl']
        }
        

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)

    def augment_kt_seqs(self, seed_change=False, **kwargs):
        """ Get augmental data, including four methods: Question mask、Interaction crop、Interaction permute、Question replac.

        * **Question mask**: replace some questions in the original history with a special token [mask], without changing their responses.
        * **Interaction crop**: create a random subset of original data.
        * **Interaction permute**: re-order interactions in a sub-sequence of the original history.
        * **Question replac**: convert original questions to easier or more difficult questions based on their responses.

        Returns: 
            tuple:
                - exer_seq_final: augmental exercise sequence.
                - cpt_seq_fianl: augmental concept sequence.
                - label_seq_final: new label after augment.
                - label_seq_flip: flipping the label for hard negative.
                - mask: new mask matrix after augment.
        """
        if seed_change:
            random.Random(self.traintpl_cfg['seed'] + 1)
            np.random.seed(self.traintpl_cfg['seed'] + 1)
        bs = kwargs['exer_seq'].size(0)
        lens = (kwargs['mask_seq'] > 0).sum(dim=1)

        # Data Augmentation
        cpt_seq_ = kwargs['cpt_unfold_seq'].clone()
        label_seq_ = kwargs['label_seq'].clone()
        exer_seq_ = kwargs['exer_seq'].clone()
        
        # Manipulate order: Question mask
        if self.modeltpl_cfg['mask_prob'] > 0:
           for b in range(bs):
                if self.datatpl_cfg['M2C_CL4KT_OP']['sequence_truncation'] == 'recent':
                    idx = random.sample(range(self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-lens[b], self.datatpl_cfg['M2C_CL4KT_OP']['window_size']), max(1, int(lens[b] * self.modeltpl_cfg['mask_prob'])))
                else:
                    idx = random.sample(range(lens[b]-1), max(1, int(lens[b] * self.modeltpl_cfg['mask_prob'])))
                for i in idx:
                        cpt_seq_[b, i] = self.n_cpt + 1
                        exer_seq_[b, i] = self.n_item + 1
        # Hard negative
        label_seq_flip = kwargs['label_seq'].clone() if self.modeltpl_cfg['hard_negative'] else label_seq_
        for b in range(bs):
            if self.datatpl_cfg['M2C_CL4KT_OP']['sequence_truncation'] == 'recent':
                idx = torch.arange(self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-lens[b], self.datatpl_cfg['M2C_CL4KT_OP']['window_size'])
            else:
                idx = torch.arange(lens[b])
            for i in idx:
                label_seq_flip[b, i] = 1 - label_seq_flip[b, i]
        # Manipulate order:Question replace
        if self.modeltpl_cfg['replace_prob'] > 0:
            for b in range(bs):
                if self.datatpl_cfg['M2C_CL4KT_OP']['sequence_truncation'] == 'recent':
                    idx = random.sample(range(self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-lens[b], self.datatpl_cfg['M2C_CL4KT_OP']['window_size']), max(1, int(lens[b] * self.modeltpl_cfg['replace_prob'])))
                else:
                    idx = random.sample(range(lens[b]-1), max(1, int(lens[b] * self.modeltpl_cfg['replace_prob'])))
                for i in idx:
                    if cpt_seq_[b, i] != self.n_cpt + 1 and i in self.datatpl_cfg['dt_info']['train_harder_cpts'] and i in self.datatpl_cfg['dt_info']['train_harder_cpts']:
                        cpt_seq_[b, i] = self.datatpl_cfg['dt_info']['train_harder_cpts'][i]  if label_seq_[b, i] == 0 else self.datatpl_cfg['dt_info']['train_harder_cpts'][i]
        # Manipulate order:Interaction permute
        if self.modeltpl_cfg['permute_prob'] > 0:
            for b in range(bs):
                reorder_seq_len = int(lens[b] * self.modeltpl_cfg['permute_prob'])
                if self.datatpl_cfg['M2C_CL4KT_OP']['sequence_truncation'] == 'recent':
                    start_pos = random.sample(range(self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-lens[b], self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-reorder_seq_len), 1)
                else:
                    start_pos = random.sample(range(lens[b]-reorder_seq_len-1), 1)

                perm = np.random.permutation(reorder_seq_len)
                exer_seq_[b, start_pos[0]:start_pos[0]+reorder_seq_len] = exer_seq_[b, start_pos[0]:start_pos[0]+reorder_seq_len][perm]
                cpt_seq_[b, start_pos[0]:start_pos[0]+reorder_seq_len] = cpt_seq_[b, start_pos[0]:start_pos[0]+reorder_seq_len][perm]
                label_seq_[b, start_pos[0]:start_pos[0]+reorder_seq_len] = label_seq_[b, start_pos[0]:start_pos[0]+reorder_seq_len][perm]
        # Manipulate order:Interaction crop
        exer_seq_final, cpt_seq_fianl, label_seq_final, mask = torch.zeros_like(exer_seq_), torch.zeros_like(cpt_seq_), torch.zeros_like(label_seq_), torch.zeros_like(kwargs['mask_seq'])
        if self.modeltpl_cfg['crop_prob'] > 0:
            for b in range(bs):
                crop_seq_len = 1 if int(lens[b] * self.modeltpl_cfg['permute_prob']) == 0 else int(lens[b] * self.modeltpl_cfg['permute_prob'])
                if self.datatpl_cfg['M2C_CL4KT_OP']['sequence_truncation'] == 'recent':
                    start_pos = random.sample(range(self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-lens[b], self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-crop_seq_len + 1), 1)
                    exer_seq_final[b, -crop_seq_len:] = exer_seq_[b, start_pos[0]:start_pos[0] + crop_seq_len]
                    cpt_seq_fianl[b, -crop_seq_len:] = cpt_seq_[b, start_pos[0]:start_pos[0] + crop_seq_len]
                    label_seq_final[b, -crop_seq_len:] = label_seq_[b, start_pos[0]:start_pos[0] + crop_seq_len]
                    mask[b, -crop_seq_len:] = 1
                else:
                    start_pos = random.sample(range(self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-lens[b], self.datatpl_cfg['M2C_CL4KT_OP']['window_size']-crop_seq_len-1), 1)
                    exer_seq_final[b, :crop_seq_len] = exer_seq_[b, start_pos[0]:start_pos[0] + crop_seq_len]
                    cpt_seq_fianl[b, :crop_seq_len] = cpt_seq_[b, start_pos[0]:start_pos[0] + crop_seq_len]
                    label_seq_final[b, :crop_seq_len] = label_seq_[b, start_pos[0]:start_pos[0] + crop_seq_len]
                    mask[b, crop_seq_len] = 1

        return exer_seq_final, cpt_seq_fianl, label_seq_final, label_seq_flip, mask

class Similarity(nn.Module):
    """ Calculate cosine similarity between data x and data y.
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class CL4KTTransformerLayer(nn.Module):
    """
    This is a Basic Block of Transformer paper. It contains one Multi-head attention object.

    Followed by layer norm and position-wise feed-forward net and dropotu layer.
    """
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(CL4KTTransformerLayer, self).__init__()
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttentionWithIndividualFeatures(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )
        # Two layer norm and two dropout layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        

    def forward(self, mask, query, key, values, apply_pos=True):
        # device = query.get_device()
        seqlen =  query.size(1)

        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask)
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        bert_mask = torch.ones_like(src_mask).bool()

        if mask == 0:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        elif mask == 1:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        else:  # mask == 2
            query2, attn = self.masked_attn_head(query, key, values, mask=bert_mask)
        
        query = query + self.dropout1((query2))  # residual connection
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query, attn


class MultiHeadAttentionWithIndividualFeatures(nn.Module):
    """
    It has projection layer for getting keys, queries, and values. 
    
    Followed by attention and a connected layer.
    """
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttentionWithIndividualFeatures, self).__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()
    
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            torch.nn.init.xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            torch.nn.init.constant_(self.k_linear.bias, 0.0)
            torch.nn.init.constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                torch.nn.init.constant_(self.q_linear.bias, 0.0)
            torch.nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, q, k, v, mask):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * num_heads * seqlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        gammas = self.gammas
        scores, attn_scores = individual_attention(
            q, k, v, self.d_k, mask, self.dropout, gammas
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # concat torch.Size([24, 200, 256])   [batch_size, seqlen, d_model]
        output = self.out_proj(concat)

        return output, attn_scores

def individual_attention(q, k, v, d_k, mask, dropout, gamma=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
        d_k
    )  # [batch_size, 8, seq_len, seq_len]
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

        # device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor)
        position_effect = position_effect.to(distcum_scores.device)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()

    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    attn_scores = scores
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_scores
