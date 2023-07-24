r"""
GKT
##########################################

Reference:
    Hiromi Nakagawa et al. "Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network" in WI 2019.

Reference Code:
    https://github.com/jhljx/GKT

"""

from ...gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .building_blocks import MLP, MLPDecoder, MLPEncoder, EraseAddGate, ScaledDotProductAttention
import scipy.sparse as sp
from .losses import VAELoss, gumbel_softmax


class GKT(GDBaseModel):
    default_cfg = {
        'emb_dim': 32,
        'hidden_dim': 32,
        'edge_type_num': 2,
        'graph_type': 'Dense',
        'dropout': 0.0,
        'MHA_cfg': {
            'atten_dim': 32,
        },
        'VAE_cfg': {
            'encoder_dim': 32,
            'decoder_dim': 32,
            'temp': 0.5,
            'bias': True,
            'factor': False,
            'var': 1.0
        }
    }

    def add_extra_data(self, **kwargs):
        self.graph = kwargs['graph']

    def build_cfg(self):
        self.graph_type = self.modeltpl_cfg['graph_type']
        self.n_stu = self.datatpl_cfg['dt_info']['stu_count']
        self.n_exer = self.datatpl_cfg['dt_info']['exer_count']
        self.edge_type_num = self.modeltpl_cfg['edge_type_num']

        self.hidden_dim = self.modeltpl_cfg['hidden_dim']
        self.emb_dim = self.modeltpl_cfg['emb_dim']

        assert self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM', 'MHA', 'VAE']

    def build_model(self):
        self.exer_emb = nn.Embedding(self.n_exer * 2, self.emb_dim)
        self.cpt_emb = nn.Embedding(self.n_exer, self.emb_dim)

        if self.graph_type in ['Dense', 'Transition', 'DKT']:
            assert  self.edge_type_num == 2
            assert self.graph is not None
            self.graph = nn.Parameter(self.graph)  # [concept_num, concept_num]
            self.graph.requires_grad = False  # fix parameter
        else:  # ['PAM', 'MHA', 'VAE']
            if self.graph_type == 'PAM':
                assert self.graph is None
                self.graph = nn.Parameter(torch.rand(self.n_exer, self.n_exer))
            elif self.graph_type == 'MHA':
                self.graph_model = MultiHeadAttention(
                    2, self.n_exer, self.emb_dim,
                    self.modeltpl_cfg['MHA_cfg']['atten_dim'],
                    dropout=self.modeltpl_cfg['dropout']
                )
            elif self.graph_type == 'VAE':
                vae_cfg = self.modeltpl_cfg['VAE_cfg']
                self.graph_model = VAE(
                    self.emb_dim, vae_cfg['encoder_dim'], 2, vae_cfg['decoder_dim'], vae_cfg['decoder_dim'], 
                    self.n_exer, edge_type_num=2, tau=vae_cfg['temp'], 
                    factor=vae_cfg['factor'], dropout=self.modeltpl_cfg['dropout'], bias=vae_cfg['bias']
                )
                self.vae_loss = VAELoss(self.n_exer, 2, prior=False, var=vae_cfg['var'])
            else:
                raise ValueError(f"unknown graph_type: {self.graph_type}")
        dropout = self.modeltpl_cfg['dropout']
        mlp_input_dim =  self.hidden_dim + self.emb_dim

        self.f_self = MLP(mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=dropout)
        self.f_neighbor_list = nn.ModuleList()
        if self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            # f_in and f_out functions
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=dropout))
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=dropout))
        else:  # ['MHA', 'VAE']
            for _ in range(self.edge_type_num):
                self.f_neighbor_list.append(MLP(2 * mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=dropout))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(self.hidden_dim, self.n_exer)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=True)
        # prediction layer
        self.pd_layer = nn.Linear(self.hidden_dim, 1, bias=True)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht, qt_mask):
        res_embedding = self.exer_emb(xt)
        concept_embedding = self.cpt_emb.weight.repeat(qt.shape[0], 1).reshape(qt.shape[0], self.n_exer, -1) # [batch_size, concept_num, embedding_dim]
        index_tuple = (torch.arange(qt_mask.sum(), device=self.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding[qt_mask]) # 只对自身进行替换，替换成另一种embebdding
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, concept_num, hidden_dim + embedding_dim]
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt, qt_mask):
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.n_exer, 1)  #[mask_num, concept_num, hidden_dim + embedding_dim]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  #[mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]
        concept_embedding, rec_embedding, z_prob = None, None, None

        if self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
            reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
            # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
            neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)
        else:  # ['MHA', 'VAE']
            concept_embedding = self.cpt_emb.weight  # [concept_num, embedding_dim]
            if self.graph_type == 'MHA':
                query = self.cpt_emb(masked_qt)
                key = concept_embedding
                att_mask = torch.ones(self.edge_type_num, mask_num, self.n_exer, device=qt.device)
                for k in range(self.edge_type_num):
                    index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
                    att_mask[k] = att_mask[k].index_put(index_tuple, torch.zeros(mask_num, device=qt.device))
                graphs = self.graph_model(masked_qt, query, key, att_mask)
            else:  # self.graph_type == 'VAE'
                sp_send, sp_rec, sp_send_t, sp_rec_t = self._get_edges(masked_qt)
                graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, sp_send, sp_rec, sp_send_t, sp_rec_t)
            neigh_features = 0
            for k in range(self.edge_type_num):
                adj = graphs[k][masked_qt, :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                if k == 0:
                    neigh_features = adj * self.f_neighbor_list[k](neigh_ht)
                else:
                    neigh_features = neigh_features + adj * self.f_neighbor_list[k](neigh_ht)
            if self.graph_type == 'MHA':
                neigh_features = 1. / self.edge_type_num * neigh_features
        # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt, qt_mask):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt, qt_mask)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.n_exer, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.pd_layer(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        return yt.gather(index=q_next.reshape(-1,1), dim=1)

    # Get edges for edge inference in VAE
    def _get_edges(self, masked_qt):
        r"""
        Parameters:
            masked_qt: qt index with -1 padding values removed
        Shape:
            masked_qt: [mask_num, ]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            rel_send: from nodes in edges which send messages to other nodes
            rel_rec:  to nodes in edges which receive messages from other nodes
        """
        mask_num = masked_qt.shape[0]
        row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
        row_arr = np.repeat(row_arr, self.n_exer, axis=1)  # [mask_num, concept_num]
        col_arr = np.arange(self.n_exer).reshape(1, -1)  # [1, concept_num]
        col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
        # add reversed edges
        new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
        new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
        row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
        col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
        data_arr = np.ones(2 * mask_num * self.n_exer)
        init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.n_exer, self.n_exer))
        init_graph.setdiag(0)  # remove self-loop edges
        row_arr, col_arr, _ = sp.find(init_graph)
        row_tensor = torch.from_numpy(row_arr).long()
        col_tensor = torch.from_numpy(col_arr).long()
        one_hot_table = torch.eye(self.n_exer, self.n_exer)
        rel_send = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_rec = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
        sp_rec, sp_send = rel_rec.to_sparse(), rel_send.to_sparse()
        sp_rec_t, sp_send_t = rel_rec.T.to_sparse(), rel_send.T.to_sparse()
        sp_send = sp_send.to(device=masked_qt.device)
        sp_rec = sp_rec.to(device=masked_qt.device)
        sp_send_t = sp_send_t.to(device=masked_qt.device)
        sp_rec_t = sp_rec_t.to(device=masked_qt.device)
        return sp_send, sp_rec, sp_send_t, sp_rec_t

    def forward(self, exer_seq, label_seq, mask_seq, **kwargs):
        batch_size, seq_len = exer_seq.shape
        features = exer_seq + label_seq.long() * self.n_exer

        ht = torch.zeros((batch_size, self.n_exer, self.hidden_dim), device=self.device)
        pred_list = []
        ec_list = []  # concept embedding list in VAE
        rec_list = []  # reconstructed embedding list in VAE
        z_prob_list = []  # probability distribution of latent variable z in VAE
        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = exer_seq[:, i]  # [batch_size]
            qt_mask = torch.eq(mask_seq[:, i], 1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(xt, qt, ht, qt_mask)  # [batch_size, concept_num, hidden_dim + embedding_dim]
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt, qt_mask)  # [batch_size, concept_num, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, exer_seq[:, i + 1], )
                pred_list.append(pred)
            ec_list.append(concept_embedding)
            rec_list.append(rec_embedding)
            z_prob_list.append(z_prob)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
        return pred_res, ec_list, rec_list, z_prob_list
    
    def get_main_loss(self, **kwargs):
        y_pd, ec_list, rec_list, z_prob_list = self(**kwargs)
        y_pd = y_pd.squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )

        if self.graph_type != 'VAE':
            return {
                'loss_main': loss,
            }
        else:
            return {
                'loss_main': loss,
                'loss_vae': self.vae_loss(ec_list, rec_list, z_prob_list)
            }


    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)[0].squeeze(dim=-1)
        # y_pd = y_pd.gather(
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
    

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    """

    def __init__(self, n_head, concept_num, input_dim, d_k, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.n_exer = concept_num
        self.d_k = d_k
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(n_head, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, attn_score, qt):
        r"""
        Parameters:
            attn_score: attention score of all queries
            qt: masked question index
        Shape:
            attn_score: [n_head, mask_num, concept_num]
            qt: [mask_num]
        Return:
            graphs: n_head types of inferred graphs
        """
        graphs = torch.zeros(self.n_head, self.n_exer, self.n_exer, device=qt.device)
        for k in range(self.n_head):
            index_tuple = (qt.long(), )
            graphs[k] = graphs[k].index_put(index_tuple, attn_score[k])  # used for calculation
            #############################
            # here, we need to detach edges when storing it into self.graphs in case memory leak!
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, attn_score[k].detach())  # used for saving and visualization
            #############################
        return graphs

    def forward(self, qt, query, key, mask=None):
        r"""
        Parameters:
            qt: masked question index
            query: answered concept embedding for a student batch
            key: concept embedding matrix
            mask: mask matrix
        Shape:
            qt: [mask_num]
            query: [mask_num, embedding_dim]
            key: [concept_num, embedding_dim]
        Return:
            graphs: n_head types of inferred graphs
        """
        d_k, n_head = self.d_k, self.n_head
        len_q, len_k = query.size(0), key.size(0)

        # Pass through the pre-attention projection: lq x (n_head *dk)
        # Separate different heads: lq x n_head x dk
        q = self.w_qs(query).view(len_q, n_head, d_k)
        k = self.w_ks(key).view(len_k, n_head, d_k)

        # Transpose for attention dot product: n_head x lq x dk
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        attn_score = self.attention(q, k, mask=mask)  # [n_head, mask_num, concept_num]
        graphs = self._get_graph(attn_score, qt)
        return graphs


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, msg_hidden_dim, msg_output_dim, concept_num, edge_type_num=2,
                 tau=0.1, factor=True, dropout=0., bias=True):
        super(VAE, self).__init__()
        self.edge_type_num = edge_type_num
        self.n_exer = concept_num
        self.tau = tau
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, factor=factor, dropout=dropout, bias=bias)
        self.decoder = MLPDecoder(input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=dropout, bias=bias)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(edge_type_num, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, edges, sp_rec, sp_send):
        r"""
        Parameters:
            edges: sampled latent graph edge weights from the probability distribution of the latent variable z
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send: one-hot encoded send-node index(sparse tensor)
        Shape:
            edges: [edge_num, edge_type_num]
            sp_rec: [edge_num, concept_num]
            sp_send: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
        """
        x_index = sp_send._indices()[1].long()  # send node index: [edge_num, ]
        y_index = sp_rec._indices()[1].long()   # receive node index [edge_num, ]
        graphs = torch.zeros(self.edge_type_num, self.n_exer, self.n_exer, device=edges.device)
        for k in range(self.edge_type_num):
            index_tuple = (x_index, y_index)
            graphs[k] = graphs[k].index_put(index_tuple, edges[:, k])  # used for calculation
            #############################
            # here, we need to detach edges when storing it into self.graphs in case memory leak!
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, edges[:, k].detach())  # used for saving and visualization
            #############################
        return graphs

    def forward(self, data, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            data: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            data: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
            output: the reconstructed data
            prob: q(z|x) distribution
        """
        logits = self.encoder(data, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [edge_num, output_dim(edge_type_num)]
        edges = gumbel_softmax(logits, tau=self.tau, dim=-1)  # [edge_num, edge_type_num]
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [concept_num, embedding_dim]
        graphs = self._get_graph(edges, sp_send, sp_rec)
        return graphs, output, prob
