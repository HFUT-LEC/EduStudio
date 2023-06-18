from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
from typing import Dict
import torch
import numpy as np
import json


class RCDDataFmt(CDInterDataFmtExtendsQ):
    def __init__(self, cfg,
                 train_dict: Dict[str, torch.tensor],
                 test_dict: Dict[str, torch.tensor],
                 feat_name2type: Dict[str, str],
                 df_Q,
                 val_dict: Dict[str, torch.tensor] = None
                 ):
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)
        self.local_map = self._construct_local_map(cfg)

    def get_extra_data(self):
        extra_dict = super().get_extra_data()
        extra_dict['local_map'] = self.local_map
        return extra_dict

    def _construct_local_map(self, cfg):
        cpt_count = self.datafmt_cfg['dt_info']['cpt_count']
        exer_count = self.datafmt_cfg['dt_info']['exer_count']
        stu_count = self.stu_count
        self.construct_cpt_relation(cfg, cpt_count)
        self.write_relation2txt(cfg)
        local_map = {
            'directed_g': self.build_graph('direct', cpt_count, cfg),
            'undirected_g': self.build_graph('undirect', cpt_count, cfg),
            'k_from_e': self.build_graph('k_from_e', cpt_count + exer_count, cfg),
            'e_from_k': self.build_graph('e_from_k', cpt_count + exer_count, cfg),
            'u_from_e': self.build_graph('u_from_e', stu_count + exer_count, cfg),
            'e_from_u': self.build_graph('e_from_u', stu_count + exer_count, cfg),
        }
        return local_map

    def build_graph(self, type, node, cfg):
        import dgl        
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        if type == 'direct':
            with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-K_Directed.txt', 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))

            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            return g
        elif type == 'undirect':
            with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-K_Undirected.txt', 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            # add edges two lists of nodes: src and dst
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            # edges are directional in DGL; make them bi-directional
            g.add_edges(dst, src)
            return g
        elif type == 'k_from_e':
            with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-k_from_e.txt', 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            # add edges two lists of nodes: src and dst
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            return g
        elif type == 'e_from_k':
            with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-e_from_k.txt', 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            # add edges two lists of nodes: src and dst
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            return g
        elif type == 'u_from_e':
            with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-u_from_e.txt', 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            # add edges two lists of nodes: src and dst
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            return g
        elif type == 'e_from_u':
            with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-e_from_u.txt', 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '').split('\t')
                    edge_list.append((int(line[0]), int(line[1])))
            # add edges two lists of nodes: src and dst
            src, dst = tuple(zip(*edge_list))
            g.add_edges(src, dst)
            return g

    def construct_cpt_relation(self, cfg, cpt_count):
        edge_dic_deno = {}
        data_file = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-log_data_all.json'
        with open(data_file, encoding='utf8') as i_f:
            data = json.load(i_f)

        # Calculate correct matrix
        knowledgeCorrect = np.zeros([cpt_count, cpt_count])
        for student in data:
            if student['log_num'] < 2:
                continue
            log = student['logs']
            for log_i in range(student['log_num'] - 1):
                if log[log_i]['score'] * log[log_i + 1]['score'] == 1:
                    for ki in log[log_i]['knowledge_code']:
                        for kj in log[log_i + 1]['knowledge_code']:
                            if ki != kj:
                                # n_{ij}
                                knowledgeCorrect[ki - 1][kj - 1] += 1.0
                                # n_{i*}, calculate the number of correctly answering i
                                if ki - 1 in edge_dic_deno.keys():
                                    edge_dic_deno[ki - 1] += 1
                                else:
                                    edge_dic_deno[ki - 1] = 1

        s = 0
        c = 0
        # Calculate transition matrix
        knowledgeDirected = np.zeros([cpt_count, cpt_count])
        for i in range(cpt_count):
            for j in range(cpt_count):
                if i != j and knowledgeCorrect[i][j] > 0:
                    knowledgeDirected[i][j] = float(knowledgeCorrect[i][j]) / edge_dic_deno[i]
                    s += knowledgeDirected[i][j]
                    c += 1
        o = np.zeros([cpt_count, cpt_count])
        min_c = 100000
        max_c = 0
        for i in range(cpt_count):
            for j in range(cpt_count):
                if knowledgeCorrect[i][j] > 0 and i != j:
                    min_c = min(min_c, knowledgeDirected[i][j])
                    max_c = max(max_c, knowledgeDirected[i][j])
        s_o = 0
        l_o = 0
        for i in range(cpt_count):
            for j in range(cpt_count):
                if knowledgeCorrect[i][j] > 0 and i != j:
                    o[i][j] = (knowledgeDirected[i][j] - min_c) / (max_c - min_c)
                    l_o += 1
                    s_o += o[i][j]
        avg = s_o / l_o  # total / count
        # avg = 0.02
        avg *= avg
        avg *= avg
        # avg is threshold
        graph = ''
        # graph, bigraph = '', ''
        edge = (o >= avg).astype(float)
        relation = np.where(edge == 1)
        # bi_relation = np.where(edge * edge.T == 1)
        for i in range(len(relation[0])):
            graph += str(relation[0][i]) + '\t' + str(relation[1][i]) + '\n'

        KG_file = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-knowledgeGraph.txt'
        with open(KG_file, 'w') as f:
            f.write(graph)
        return relation


    def write_relation2txt(self, cfg):
        K_Directed = ''
        K_Undirected = ''
        edge = []
        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-knowledgeGraph.txt', 'r') as f:
            for i in f.readlines():
                i = i.replace('\n', '').split('\t')
                src = i[0]
                tar = i[1]
                edge.append((src, tar))
        visit = []
        for e in edge:
            if e not in visit:
                if (e[1], e[0]) in edge:
                    K_Undirected += str(e[0] + '\t' + e[1] + '\n')
                    visit.append(e)
                    visit.append((e[1], e[0]))
                else:
                    K_Directed += str(e[0] + '\t' + e[1] + '\n')
                    visit.append(e)

        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-K_Directed.txt', 'w') as f:
            f.write(K_Directed)
        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-K_Undirected.txt', 'w') as f:
            f.write(K_Undirected)

        # obtain e - k
        e_from_k, k_from_e = '',''
        cpt = self.df_Q['cpt_seq']
        exer = self.df_Q['exer_id']
        for i in range(len(cpt)):
            for each in cpt[i]:
                e_from_k += str(exer[i]) + '\t' + str(each) + '\n'
                k_from_e += str(each) +'\t'+ str(exer[i])  + '\n'

        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-e_from_k.txt', 'w') as f:
            f.write(e_from_k)
        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-k_from_e.txt', 'w') as f:
            f.write(k_from_e)


        # obtain u - e
        u_from_e, e_from_u = '', ''
        for i in range(len(self.train_dict)):
            u_from_e += str(self.train_dict['stu_id'][i].item()) + '\t' + str(self.train_dict['exer_id'][i].item()) + '\n'
            e_from_u += str(self.train_dict['exer_id'][i].item()) + '\t' + str(self.train_dict['stu_id'][i].item()) + '\n'


        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-u_from_e.txt', 'w') as f:
            f.write(u_from_e)
        with open(f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-e_from_u.txt', 'w') as f:
            f.write(e_from_u)
