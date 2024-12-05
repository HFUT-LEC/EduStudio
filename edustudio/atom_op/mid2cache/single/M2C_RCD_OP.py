from ..common import BaseMid2Cache
import numpy as np
import importlib


class M2C_RCD_OP(BaseMid2Cache):
    default_cfg = {}

    def process(self, **kwargs):        
        self.dgl = importlib.import_module("dgl")
        k_e, e_k = self.build_k_e(kwargs)
        u_e_list, e_u_list = self.build_u_e(kwargs)
        local_map = {
            'directed_g': self.build_cpt_directd(kwargs),
            'undirected_g': self.build_cpt_undirected(kwargs),
            'k_from_e': k_e,
            'e_from_k': e_k,
            'u_from_e_list': u_e_list,
            'e_from_u_list': e_u_list,
        }
        kwargs['local_map'] = local_map
        return kwargs

    def build_cpt_undirected(self, kwargs):
        cpt_count =  kwargs['dt_info']['cpt_count']
        cpt_dep_mat = kwargs['cpt_dep_mat']
        cpt_dep_mat_undirect = ((cpt_dep_mat + cpt_dep_mat.T) == 2).astype(np.int64)
        # undirected (only prerequisite)
        g_undirected = self.dgl.graph(np.argwhere(cpt_dep_mat_undirect == 1).tolist(), num_nodes=cpt_count)
        return g_undirected
    
    def build_cpt_directd(self, kwargs):
        cpt_count =  kwargs['dt_info']['cpt_count']
        cpt_dep_mat = kwargs['cpt_dep_mat']
        # directed (prerequisite + similarity)
        g_directed =self.dgl.graph(np.argwhere(cpt_dep_mat == 1).tolist(), num_nodes=cpt_count)
        return g_directed
    
    def build_k_e(self, kwargs):
        cpt_count = kwargs['dt_info']['cpt_count']
        exer_count = kwargs['dt_info']['exer_count']
        df_exer = kwargs['df_exer']
        
        edges = df_exer[['exer_id:token','cpt_seq:token_seq']].explode('cpt_seq:token_seq').to_numpy()
        edges[:, 1] += exer_count

        k_e = self.dgl.graph(edges.tolist(), num_nodes=cpt_count + exer_count)
        e_k = self.dgl.graph(edges[:,[1,0]].tolist(), num_nodes=cpt_count + exer_count)
        return k_e, e_k

    def build_u_e(self, kwargs):
        stu_count = kwargs['dt_info']['stu_count']
        exer_count = kwargs['dt_info']['exer_count']
        df_train_folds = kwargs['df_train_folds']

        u_from_e_list= []
        e_from_u_list = []
        for train_df in df_train_folds:
            stu_id = train_df['stu_id:token'].to_numpy() + exer_count
            exer_id = train_df['exer_id:token'].to_numpy()
            u_e = self.dgl.graph(np.vstack([exer_id, stu_id]).T.tolist(), num_nodes=stu_count + exer_count)
            e_u = self.dgl.graph(np.vstack([stu_id, exer_id]).T.tolist(), num_nodes=stu_count + exer_count)
            u_from_e_list.append(u_e)
            e_from_u_list.append(e_u)
        return u_from_e_list, e_from_u_list
