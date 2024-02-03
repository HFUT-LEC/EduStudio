from ..common import EduDataTPL
import numpy as np
import torch


class GKTDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ["M2C_KCAsExer", 'M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats', 'M2C_RandomDataSplit4KT'],
    }

    def process_load_data_from_middata(self):
        super().process_load_data_from_middata()
        concept_num = self.final_kwargs['dt_info']['exer_count']
        if self.modeltpl_cfg['graph_type'] == 'Dense':
            self.final_kwargs['gkt_graph'] = self.build_dense_graph(node_num=concept_num)
        elif self.modeltpl_cfg['graph_type'] == 'Transition':
            self.final_kwargs['gkt_graph_list'] = []
            for train_dict in self.final_kwargs['df_train_folds']:
                self.final_kwargs['gkt_graph_list'].append(
                    self.build_transition_graph(
                        exer_seq=train_dict['exer_seq:token_seq'],
                        mask_seq=train_dict['mask_seq:token_seq'],
                        concept_num=concept_num
                    )
                )   
        elif self.modeltpl_cfg['graph_type'] == 'DKT':
            self.final_kwargs['gkt_graph'] = self.build_dkt_graph(
                f"{self.frame_cfg.data_folder_path}/dkt_graph.txt", 
                concept_num=concept_num
            )
        elif self.modeltpl_cfg['graph_type'] == 'PAM':
            self.final_kwargs['gkt_graph'] = None
        elif self.modeltpl_cfg['graph_type'] ==  'MHA':
            self.final_kwargs['gkt_graph'] = None
        elif self.modeltpl_cfg['graph_type'] ==  'VAE':
            self.final_kwargs['gkt_graph'] = None
        else:
            raise ValueError(f"unknown graph_type: {self.modeltpl_cfg['graph_type']}")

    def get_extra_data(self, **kwargs):
        dic = super().get_extra_data(**kwargs)
        dic['graph'] = self.final_kwargs['gkt_graph']
        return dic
    
    @staticmethod
    def build_transition_graph(exer_seq, mask_seq, concept_num):
        graph = np.zeros((concept_num, concept_num))
        for exers, masks in zip(exer_seq, mask_seq):
            for idx in range(len(exers) - 1):
                if masks[idx+1] == 0: break
                graph[exers[idx], exers[idx+1]] += 1
        np.fill_diagonal(graph, 0)
        # row normalization
        rowsum = np.array(graph.sum(1))
        def inv(x):
            if x == 0:
                return x
            return 1. / x
        inv_func = np.vectorize(inv)
        r_inv = inv_func(rowsum).flatten()
        r_mat_inv = np.diag(r_inv)
        graph = r_mat_inv.dot(graph)
        # covert to tensor
        graph = torch.from_numpy(graph).float()
        return graph
    
    @staticmethod
    def build_dense_graph(node_num):
        graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
        np.fill_diagonal(graph, 0)
        graph = torch.from_numpy(graph).float()
        return graph

    @staticmethod
    def build_dkt_graph(file_path, concept_num):
        graph = np.loadtxt(file_path)
        assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
        graph = torch.from_numpy(graph).float()
        return graph

    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        if self.modeltpl_cfg['graph_type'] == 'Transition':
            self.final_kwargs['gkt_graph'] = self.final_kwargs['gkt_graph_list'][fold_id]
