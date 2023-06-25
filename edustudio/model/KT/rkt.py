r"""
RKT
##########################################

Reference:
    Shalini Pandey et al. "RKT: Relation-Aware Self-Attention for Knowledge Tracing." in CIKM 2020.

Reference Code:
    https://github.com/shalini1194/RKT/blob/master/RKT/model_rkt.py

"""
import math
import copy
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from torch.nn.init import xavier_uniform_, constant_

class RKT(GDBaseModel):
    r"""
   RKT

   default_cfg:
      'embed_size': 200  # dimension of embedding
      'drop_prob': 0.2      # dropout rate
      'num_attn_layers': 1        # number of attention layers
        'num_heads': 5         # number of parallel attention heads
        'encode_pos':False            # if True, use relative position embeddings
         'max_pos': 10          # number of position embeddings to use
   """
    default_cfg = {
        'embed_size': 200,
        'num_attn_layers': 1,
        'num_heads': 5,
        'encode_pos': False,
        'max_pos': 10,
        'drop_prob':0.2,
    }


    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.embed_size = self.modeltpl_cfg['embed_size']
        self.num_attn_layers = self.modeltpl_cfg['num_attn_layers']
        self.num_heads = self.modeltpl_cfg['num_heads']
        self.encode_pos = self.modeltpl_cfg['encode_pos']
        self.max_pos = self.modeltpl_cfg['max_pos']
        self.drop_prob = self.modeltpl_cfg['drop_prob']
        
    def build_model(self):
        self.item_embeds = nn.Embedding(self.n_item, self.embed_size, padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(self.max_pos, self.embed_size // self.num_heads)
        self.pos_value_embeds = nn.Embedding(self.max_pos, self.embed_size // self.num_heads)

        self.lin_in = nn.Linear(2 * self.embed_size, self.embed_size)
        self.attn_layers = clone(MultiHeadedAttention(self.embed_size, self.num_heads, self.drop_prob, self.device), self.num_attn_layers)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.lin_out = nn.Linear(self.embed_size, 1)
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))
        # self.pro_pro_dense = self.get_corr_data()

    def get_inputs(self, item_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        # skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, item_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids):
        item_ids = self.item_embeds(item_ids)
        # skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids], dim=-1)
        return query

    def add_extra_data(self, **kwargs):
        self.pro_pro_dense = kwargs['pro_pro_dense']

    def forward(self, exer_seq, label_seq,  start_timestamp_seq, **kwargs):
        time = computeRePos(start_timestamp_seq)
        item_inputs = torch.cat((torch.zeros((exer_seq.shape[0], 1)).to(self.device), exer_seq[:, :-1]), dim=1)
        label_inputs = torch.cat((torch.zeros((exer_seq.shape[0], 1)).to(self.device), label_seq[:, :-1]), dim=1)
        item_inputs = item_inputs.long().cpu()
        exer_seq = exer_seq.cpu()
        rel = self.pro_pro_dense[
            (exer_seq).unsqueeze(1).repeat(1, exer_seq.shape[-1], 1), (item_inputs).unsqueeze(-1).repeat(1, 1,
                                                                                                                 item_inputs.shape[
                                                                                                                 -1])]
        item_inputs = item_inputs.to(self.device)
        exer_seq = exer_seq.to(self.device)
        rel = torch.tensor(rel).to(self.device)

        inputs = self.get_inputs(item_inputs, label_inputs)

        inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(exer_seq)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.to(self.device)
        outputs, attn = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, time, self.encode_pos,
                                            self.pos_key_embeds, self.pos_value_embeds, mask)
        outputs = self.dropout(outputs)
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, self.encode_pos, time,
                               self.pos_key_embeds,
                               self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return torch.sigmoid(self.lin_out(outputs))

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
            'y_pd': y_pd.squeeze(1),
            'y_gt': y_gt
        }

    def get_main_loss(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd.squeeze(1), target=y_gt
        )
        return {
            'loss_main': loss,
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)



def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

def computeRePos(time_seq):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix= (torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1)) - \
                 torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1))))

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size,size,size))


    return (time_matrix)

def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def attention(query, key, value, rel, l1, l2, timestamp, mask=None, dropout=None, device='cpu'):
    """Compute scaled dot product attention.
    """
    rel = rel * mask.to(torch.float) # future masking of correlation matrix.
    rel_attn = rel.masked_fill(rel == 0, -10000)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

        time_stamp= torch.exp(-torch.abs(timestamp.float()))
        #
        time_stamp=time_stamp.masked_fill(mask,-np.inf)


    prob_attn = F.softmax(scores, dim=-1)
    time_attn = F.softmax(time_stamp,dim=-1)
    prob_attn = (1-l2)*prob_attn+l2*time_attn
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)

    prob_attn = (1-l1)*prob_attn + (l1)*rel_attn
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, rel, l1, l2, pos_key_embeds, pos_value_embeds, mask=None, dropout=None, device='cpu'):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings

    scores = torch.matmul(query, key.transpose(-2, -1))

    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.to(device)
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob, device):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.device = device
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, rel, l1, l2, timestamp, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)
        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, rel, l1, l2, timestamp, pos_key_embeds, pos_value_embeds,  mask, self.dropout, self.device)
        else:
            out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, mask, self.dropout, self.device)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn

