r"""
RCD
##################################
Reference:
    Weibo Gao et al. "RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems." in SIGIR 2021.

Reference Code:
    https://github.com/bigdata-ustc/RCD
"""


from ..gd_basemodel import GDBaseModel
import torch.nn as nn
from ..utils.components import PosMLP
import torch
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class Fusion(nn.Module):
    def __init__(self, n_user, n_item, n_cpt, local_map, device):
        self.device = device
        self.knowledge_dim = n_cpt
        self.exer_n = n_item
        self.emb_num = n_user
        self.stu_dim = self.knowledge_dim

        # graph structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)

        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, n_cpt, n_cpt)
        self.undirected_gat = GraphLayer(self.undirected_g, n_cpt, n_cpt)

        self.k_from_e = GraphLayer(self.k_from_e, n_cpt, n_cpt)  # src: e
        self.e_from_k = GraphLayer(self.e_from_k, n_cpt, n_cpt)  # src: k

        self.u_from_e = GraphLayer(self.u_from_e, n_cpt, n_cpt)  # src: e
        self.e_from_u = GraphLayer(self.e_from_u, n_cpt, n_cpt)  # src: u

        self.k_attn_fc1 = nn.Linear(2 * n_cpt, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * n_cpt, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * n_cpt, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * n_cpt, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * n_cpt, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        k_directed = self.directed_gat(kn_emb)
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)

        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.u_from_e(e_u_graph)
        e_from_u_graph = self.e_from_u(e_u_graph)

        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        D = k_from_e_graph[self.exer_n:]
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)
        score1 = self.k_attn_fc1(concat_c_1)
        score2 = self.k_attn_fc2(concat_c_2)
        score3 = self.k_attn_fc3(concat_c_3)
        score = F.softmax(torch.cat([torch.cat([score1, score2], dim=1), score3], dim=1),
                          dim=1)  # dim = 1, 按行SoftMax, 行和为1
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C + score[:, 2].unsqueeze(1) * D

        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0: self.exer_n]
        C = e_from_u_graph[0: self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)  # dim = 1, 按行SoftMax, 行和为1
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_n:]

        return kn_emb, exer_emb, all_stu_emb


class RCD(GDBaseModel):
    default_cfg = {
        'prednet_len1': 512,
        'prednet_len2': 256
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def add_extra_data(self, **kwargs):
        self.local_map = kwargs.pop('local_map')
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    def build_model(self):
        self.knowledge_dim = self.n_cpt
        self.exer_n = self.n_item
        self.emb_num = self.n_user
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = self.modeltpl_cfg['prednet_len1'], self.modeltpl_cfg['prednet_len2']
        self.directed_g = self.local_map['directed_g'].to(self.device)
        self.undirected_g = self.local_map['undirected_g'].to(self.device)
        self.k_from_e = self.local_map['k_from_e'].to(self.device)
        self.e_from_k = self.local_map['e_from_k'].to(self.device)
        self.u_from_e = self.local_map['u_from_e'].to(self.device)
        self.e_from_u = self.local_map['e_from_u'].to(self.device)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        self.student_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.knowledge_emb = nn.Embedding(self.n_cpt, self.n_cpt)
        self.exercise_emb = nn.Embedding(self.n_item, self.n_cpt)

        self.FusionLayer1 = Fusion(self.n_user, self.n_item, self.n_cpt, self.local_map, self.device)
        self.FusionLayer2 = Fusion(self.n_user, self.n_item, self.n_cpt, self.local_map, self.device)

        self.prednet_full1 = nn.Linear(2 * self.n_cpt, self.n_cpt, bias=False)
        self.prednet_full2 = nn.Linear(2 * self.n_cpt, self.n_cpt, bias=False)
        self.prednet_full3 = nn.Linear(1 * self.n_cpt, 1)


    def forward(self, stu_id, exer_id, **kwargs):
        # before prednet
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        # get batch student data
        batch_stu_emb = all_stu_emb2[stu_id]  # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0],
                                                                      kn_emb2.shape[1])

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * self.Q_mat[exer_id].unsqueeze(2), dim=1)
        count_of_concept = torch.sum(self.Q_mat[exer_id], dim=1).unsqueeze(1)
        pd = sum_out / count_of_concept
        return pd

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

        # output_1 = self(stu_id, exer_id, Q_mat)
        # output_0 = torch.ones(output_1.size()).to(self.device) - output_1
        # output = torch.cat((output_0, output_1), 1)
        # pd = torch.log(output+1e-10).flatten()

        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    # def get_stu_status(self, stu_id=None):
    #     if stu_id is not None:
    #         return self.student_emb(stu_id)
    #     else:
    #         return self.student_emb.weight
