r"""
HierCDF
##########################################

Reference:
    Jiatong Li et al. "HierCDF: A Bayesian Network-Based Hierarchical Cognitive Diagnosis Framework" in KDD 2022.

Reference Code:
    https://github.com/CSLiJT/HCD-code

"""


import torch
import torch.nn as nn
import networkx as nx
from ..gd_basemodel import GDBaseModel
import numpy as np 
import pandas as pd 
from ..utils.components import PosLinear
import torch.nn.functional as F

def irt2pl(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return 1 / (1 + torch.exp(-1.7*item_offset*(user_emb - item_emb) ))

def mirt2pl(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return 1 / (1 + torch.exp(- torch.sum(torch.mul(user_emb, item_emb), axis=1).reshape(-1,1) + item_offset))

def sigmoid_dot(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return torch.sigmoid(torch.sum(torch.mul(user_emb, item_emb), axis = -1)).reshape(-1,1)

def dot(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return torch.sum(torch.mul(user_emb, item_emb), axis = -1).reshape(-1,1)

itf_dict = {
    'irt': irt2pl,
    'mirt': mirt2pl,
    'mf': dot, 
    'sigmoid-mf': sigmoid_dot
}

class HierCDF(GDBaseModel):
    default_cfg = {
        'itf_type': 'mirt',
        'hidden_dim': 1,
        'lambda': 0.001, 
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.hidden_dim = self.modeltpl_cfg['hidden_dim']
        self.itf_type = self.modeltpl_cfg['itf_type']

        self.set_itf(self.itf_type)

    def build_model(self):
        # the conditional mastery degree when parent is mastered
        self.condi_p = nn.Embedding(self.n_user, self.know_graph.shape[0])

        # the conditional mastery degree when parent is non-mastered
        self.condi_n = nn.Embedding(self.n_user, self.know_graph.shape[0])

        # the priori mastery degree of parent
        self.priori = nn.Embedding(self.n_user, self.n_cpt)

        # item representation
        self.item_diff = nn.Embedding(self.n_item, self.n_cpt)
        self.item_disc = nn.Embedding(self.n_item, 1)

        # embedding transformation
        self.user_contract = PosLinear(self.n_cpt, self.hidden_dim)
        self.item_contract = PosLinear(self.n_cpt, self.hidden_dim)

        # Neural Interaction Module (used only in ncd)
        self.cross_layer1 = PosLinear(self.hidden_dim,max(int(self.hidden_dim/2),1))
        self.cross_layer2 = PosLinear(max(int(self.hidden_dim/2),1),1)

    def add_extra_data(self, df_cpt_relation, **kwargs):
        self.know_graph = df_cpt_relation
        self.know_edge = nx.DiGraph()  #nx.DiGraph(know_graph.values.tolist())
        for k in range(self.n_cpt):
            self.know_edge.add_node(k)
        for edge in df_cpt_relation[['cpt_head', 'cpt_tail']].to_numpy():
            self.know_edge.add_edge(edge[0],edge[1])
        self.topo_order = list(nx.topological_sort(self.know_edge))
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    def ncd(self, user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
        input_vec = (user_emb-item_emb)*item_offset
        x_vec=torch.sigmoid(self.cross_layer1(input_vec))
        x_vec=torch.sigmoid(self.cross_layer2(x_vec))
        return x_vec
    
    def set_itf(self, itf_type):
        self.itf_type = itf_type
        self.itf = itf_dict.get(itf_type, self.ncd)

    def get_posterior(self, user_ids: torch.LongTensor):
        n_batch = user_ids.shape[0]
        posterior = torch.rand(n_batch, self.n_cpt).to(self.device)
        batch_priori = torch.sigmoid(self.priori(user_ids))
        batch_condi_p = torch.sigmoid(self.condi_p(user_ids))
        batch_condi_n = torch.sigmoid(self.condi_n(user_ids))
        
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)

            # for each knowledge k, do:
            if len_p == 0:
                priori = batch_priori[:,k]
                posterior[:,k] = priori.reshape(-1)
                continue

            # format of masks
            tpl = '{0:0%db}'%(len_p)
            # number of parent master condition
            n_condi = 2 ** len_p

            priori = posterior[:,predecessors]


            pred_idx = self.know_graph[self.know_graph['cpt_tail'] == k].sort_values(by='cpt_head').index
            condi_p = torch.pow(batch_condi_p[:,pred_idx],1/len_p)
            condi_n = torch.pow(batch_condi_n[:,pred_idx],1/len_p)
            
            margin_p = condi_p * priori
            margin_n = condi_n * (1.0-priori)

            posterior_k = torch.zeros((1,n_batch)).to(self.device)

            for idx in range(n_condi):
                # for each parent mastery condition, do:
                mask = tpl.format(idx)
                mask = torch.Tensor(np.array(list(mask)).astype(int)).to(self.device)

                margin = mask * margin_p + (1-mask) * margin_n
                margin = torch.prod(margin, dim = 1).unsqueeze(dim = 0)

                posterior_k = torch.cat([posterior_k, margin], dim = 0)
            posterior_k = (torch.sum(posterior_k, dim = 0)).squeeze()
            
            posterior[:,k] = posterior_k.reshape(-1)
        return posterior
    
    def get_condi_p(self,user_ids: torch.LongTensor, device = 'cpu')->torch.Tensor:
        n_batch = user_ids.shape[0]
        result_tensor = torch.rand(n_batch, self.n_cpt).to(device)
        batch_priori = torch.sigmoid(self.priori(user_ids))
        batch_condi_p = torch.sigmoid(self.condi_p(user_ids))
        
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:,k]
                result_tensor[:,k] = priori.reshape(-1)
                continue
            pred_idx = self.know_graph[self.know_graph['cpt_tail'] == k].sort_values(by='cpt_head').index
            condi_p = torch.pow(batch_condi_p[:,pred_idx],1/len_p)
            result_tensor[:,k] = torch.prod(condi_p, dim=1).reshape(-1)
        
        return result_tensor

    def get_condi_n(self,user_ids: torch.LongTensor, device = 'cpu')->torch.Tensor:
        n_batch = user_ids.shape[0]
        result_tensor = torch.rand(n_batch, self.n_cpt).to(device)
        batch_priori = torch.sigmoid(self.priori(user_ids))
        batch_condi_n = torch.sigmoid(self.condi_n(user_ids))
        
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:,k]
                result_tensor[:,k] = priori.reshape(-1)
                continue
            pred_idx = self.know_graph[self.know_graph['cpt_tail'] == k].sort_values(by='cpt_head').index
            condi_n = torch.pow(batch_condi_n[:,pred_idx],1/len_p)
            result_tensor[:,k] = torch.prod(condi_n, dim=1).reshape(-1)
        
        return result_tensor

    def forward(self, stu_id: torch.LongTensor, exer_id: torch.LongTensor):
        Q_mat = self.Q_mat[exer_id]
        user_mastery = self.get_posterior(stu_id)
        item_diff = torch.sigmoid(self.item_diff(exer_id))
        item_disc = torch.sigmoid(self.item_disc(exer_id))

        user_factor = torch.tanh(self.user_contract(user_mastery * Q_mat))
        item_factor = torch.sigmoid(self.item_contract(item_diff * Q_mat))
        
        output = self.itf(user_factor, item_factor, item_disc)

        return output


    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd = self(stu_id, exer_id).flatten()
        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }

    def get_J_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        return {
            'loss_J': self.modeltpl_cfg['lambda'] * \
                torch.sum(torch.relu(self.condi_n(stu_id)-self.condi_p(stu_id)))
        }
    
    def get_loss_dict(self, **kwargs):
        ret_dict = self.get_main_loss(**kwargs)
        ret_dict.update(self.get_J_loss(**kwargs))
        return ret_dict
    
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.get_posterior(stu_id)
        else:
            stu_id = torch.arange(self.n_user).to(self.device)
            return self.get_posterior(stu_id)
