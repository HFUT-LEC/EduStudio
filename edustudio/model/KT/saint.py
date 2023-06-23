r"""
SAINT
##########################################

Reference:
     Youngduck Choi et al. "Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing" in L-AT-S 2020.

Reference Code:
    https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/saint.py

"""
import copy

import math
import pandas as pd
from torch.nn import Dropout

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, constant_

class SAINT(GDBaseModel):
    r"""
    SAINT

    default_cfg:
       'emb_size': 256          # dimension of embedding
        'num_attn_heads': 8     # number of parallel attention heads
        'dropout_rate': 0.2     # dropout rate
        'n_blocks':4          # number of Encoder_blocks
    """
    default_cfg = {
        'emb_size': 256,
        'num_attn_heads': 8,
        'dropout_rate': 0.2,
        'n_blocks':4,
    }


    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.num_q = self.datatpl_cfg['dt_info']['exer_count']
        self.num_c = self.datatpl_cfg['dt_info']['cpt_count']
        self.window_size = self.datatpl_cfg['dt_info']['real_window_size']
        self.dropout_r = self.modeltpl_cfg['dropout_rate']
        self.num_attn_heads = self.modeltpl_cfg['num_attn_heads']
        self.emb_size = self.modeltpl_cfg['emb_size']
        self.n_blocks = self.modeltpl_cfg['n_blocks']
        
    def build_model(self):
        self.embd_pos = nn.Embedding(self.window_size, embedding_dim=self.emb_size)
        self.encoder = get_clones(Encoder_block(self.emb_size, self.num_attn_heads, self.num_q, self.num_c, self.window_size, self.dropout_r, self.device),
                                      self.n_blocks)

        self.decoder = get_clones(Decoder_block(self.emb_size, 2, self.num_attn_heads, self.window_size, self.dropout_r, self.device), self.n_blocks)

        self.dropout = Dropout(self.dropout_r)
        self.out = nn.Linear(in_features=self.emb_size, out_features=1)


    def forward(self, exer_seq, label_seq, cpt_unfold_seq, **kwargs):
        label_seq = label_seq.type(torch.int64)
        if self.num_q > 0:
            in_pos = torch.arange(exer_seq.shape[1]).unsqueeze(0).to(self.device)
        else:
            in_pos = torch.arange(cpt_unfold_seq.shape[1]).unsqueeze(0).to(self.device)
        in_pos = self.embd_pos(in_pos)
        first_block = True
        for i in range(self.n_blocks):
            if i >= 1:
                first_block = False
            exer_seq = self.encoder[i](exer_seq, cpt_unfold_seq, in_pos, first_block=first_block)
            cpt_unfold_seq = exer_seq
        first_block = True
        for i in range(self.n_blocks):
            if i >= 1:
                first_block = False
            label_seq = self.decoder[i](label_seq, in_pos, en_out=exer_seq, first_block=first_block)
        ## Output layer
        res = self.out(self.dropout(label_seq))
        res = torch.sigmoid(res).squeeze(-1)
        return res

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
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)


def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class transformer_FFN(nn.Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(device, seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)
class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout,
                  device, emb_path="", pretrain_dim=768):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex,
                                            embedding_dim=dim_model)  # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = nn.Embedding.from_pretrained(embs)
                self.linear = nn.Linear(pretrain_dim, dim_model)
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim=dim_model)
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):

        ##
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1, 0, 2)  # (n,b,d)  # print('pre multi', out.shape)

        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)  # Layer norm
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout1(out)
        out = out + skip_out  # skip connection

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm2(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out  # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout, device):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        self.embd_res = nn.Embedding(total_res + 1,
                                     embedding_dim=dim_model)  # response embedding, include a start token
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de,
                                               dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de,
                                               dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en = transformer_FFN(dim_model, dropout)  # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, in_res, in_pos, en_out, first_block=True):

        ##
        if first_block:
            in_in = self.embd_res(in_res)

            # combining the embedings
            out = in_in + in_pos  # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1, 0, 2)  # (n,b,d)# print('pre multi', out.shape)
        n, _, _ = out.shape

        # Multihead attention M1
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout1(out)
        out = skip_out + out  # skip connection

        # Multihead attention M2
        en_out = en_out.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm3(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout3(out)
        out = out + skip_out  # skip connection

        return out