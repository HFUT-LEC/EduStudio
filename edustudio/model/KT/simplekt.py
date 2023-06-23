import math

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, constant_

class SimpleKT(GDBaseModel):
    default_cfg = {
        'l2': 1e-5,
        'kq_same': 1,
        'dropout_rate': 0.05,
        'separate_qa': False,
        'd_model': 256,
        'n_blocks': 1,
        'final_fc_dim': 512,
        'n_heads': 8,
        'd_ff': 2048,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

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
        self.seq_len = self.datatpl_cfg['dt_info']['real_window_size']
        self.device = self.traintpl_cfg['device']

    def build_model(self):
        embed_l = self.d_model
        self.n_pid = self.n_item
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_blocks=self.n_blocks,
                                  n_heads=self.n_heads, dropout=self.dropout,
                                  d_model=self.d_model,
                                  d_ff=self.d_ff, kq_same=self.kq_same, seq_len=self.seq_len, device=self.device)

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



        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        pre = torch.sigmoid(output).squeeze(-1)
        return pre
        # if self.training:
        #     return pre, c_reg_loss
        # else:
        #     return pre

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
        # y_pd, c_reg_loss = self(**kwargs)
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss,
            # 'loss_reg': c_reg_loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model,
                 d_ff, n_heads, dropout, kq_same, seq_len, device):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.device = device


        self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, device = device)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y,
                      apply_pos=True)  # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, device):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        self.device = device
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, device=device)

        # Two layer norm layer and two droput layer
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
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(self.device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask,
                zero_pad=True)  # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, device, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.device = device

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

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
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, self.device)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, device):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32).to(device)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # ( 1,seq,  Feature)