r"""
AKT
##########################################

Reference:
    Aritra Ghosh et al. "Context-Aware Attentive Knowledge Tracing" in KDD 2020.

Reference Code:
    https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/akt.py
    https://github.com/arghosh/AKT

"""
import math

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, constant_

class AKT(GDBaseModel):
    r"""
    AKT

    default_cfg:
        'l2': 1e-5              # weight of l2 regularization
        'kq_same': 1              # if k and q share the same structure, 1 denotes yes,0 denotes no,used in MultiHeadAttention
        'dropout_rate': 0.05      # dropout rate
        'separate_qa': False      # Whether to separate questions and answers
        'd_model': 256      # dimension of attention input/output
        'n_blocks':1      # number of stacked blocks in the attention
        'final_fc_dim': 512      # dimension of final fully connected net before prediction
        'n_heads': 8      # number of heads. n_heads*d_feature = d_model
        'd_ff': 2048      # dimension for fully conntected net inside the basic block
    """

    default_cfg = {
        'l2': 1e-5,
        'kq_same': 1,
        'dropout_rate': 0.05,
        'separate_qa': False,
        'd_model': 256,
        'n_blocks':1,
        'final_fc_dim': 512,
        'n_heads': 8,
        'd_ff': 2048,
    }
    def _init_params(self):
        super()._init_params()
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                constant_(p, 0.)
    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_question = self.datatpl_cfg['dt_info']['cpt_count']
        self.dropout = self.modeltpl_cfg['dropout_rate']
        self.kq_same = self.modeltpl_cfg['kq_same']
        self.l2 = self.modeltpl_cfg['l2']
        self.separate_qa = self.modeltpl_cfg['separate_qa']
        self.d_model = self.modeltpl_cfg['d_model']
        self.n_blocks = self.modeltpl_cfg['n_blocks']
        self.final_fc_dim = self.modeltpl_cfg['final_fc_dim']
        self.n_heads = self.modeltpl_cfg['n_heads']
        self.d_ff = self.modeltpl_cfg['d_ff']
        
    def build_model(self):
        embed_l = self.d_model
        self.n_pid = self.n_item
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question + 1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(self.n_blocks, self.d_model, self.d_model // self.n_heads,
                                  self.d_ff, self.n_heads, self.dropout, self.kq_same, self.device)

        self.out = nn.Sequential(
            nn.Linear(self.d_model + embed_l, self.final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )


    def forward(self, exer_seq, label_seq, cpt_unfold_seq, **kwargs):
        # Batch First
        # q_embed_data = self.q_embed(q_data)
        q_embed_data = self.q_embed(cpt_unfold_seq)
        if self.separate_qa:
            # qa_embed_data = self.qa_embed(qa_data)
            qa_data = cpt_unfold_seq + label_seq.long() * self.n_question
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = label_seq.long()
            qa_embed_data = self.qa_embed(qa_data) + q_embed_data

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(cpt_unfold_seq)
            pid_embed_data = self.difficult_param(exer_seq)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.
        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        pre = torch.sigmoid(output).squeeze(-1)
        if self.training:
            return pre, c_reg_loss
        else:
            return pre

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
        y_pd, c_reg_loss = self(**kwargs)
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss,
            'loss_reg': c_reg_loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)

class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same, device):
        super(Architecture, self).__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model, d_feature, d_ff, n_heads, dropout, kq_same, device)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model, d_feature, d_ff, n_heads, dropout, kq_same, device)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        x = q_embed_data
        y = qa_embed_data

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, device):
        super(TransformerLayer, self).__init__()
        """
        This is a Basic Block of Transformer paper. It contains one Multi-head attention object. Followed by layer
        norm and position wise feedforward net and dropout layer.
        """
        self.device = device
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout, device, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module).
                    It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            values: In transformer paper,
                    it is the input for encoder and encoded output for decoder (in masked attention part)
        Output:
            query: Input gets changed over the layer and returned.
        """
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(self.device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, device, kq_same,  bias=True):
        super(MultiHeadAttention, self).__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.device = device
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
        xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, self.device, self.gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, device, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill(mask == 0, -1e23)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output