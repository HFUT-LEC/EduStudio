import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio
from edustudio.atom_op.mid2cache.CD import M2C_FilterRecords4CD

import networkx as nx
import pandas as pd

class M2C_MyFilterRecords4CD(M2C_FilterRecords4CD):
    
    @staticmethod
    def build_DAG(edges):
        graph = nx.DiGraph()
        for edge in edges:
            graph.add_edge(*edge)
            if not nx.is_directed_acyclic_graph(graph):
                graph.remove_edge(*edge)
        return graph.edges
    
    def process(self, **kwargs):
        selected_items = kwargs['df_exer']['exer_id:token'].to_numpy()
        # kwargs['df_cpt_relation'] = kwargs['df_cpt_relation'].drop_duplicates(["cpt_head:token", "cpt_tail:token"])
        edges = kwargs['df_cpt_relation'][['cpt_head:token', 'cpt_tail:token']].to_numpy()
        edges = list(self.build_DAG(edges))
        kwargs['df_cpt_relation'] = pd.DataFrame({"cpt_head:token": [i[0] for i in edges], "cpt_tail:token": [i[1] for i in edges]})
        selected_items = kwargs['df_cpt_relation'].to_numpy().flatten()
        df = kwargs['df']
        df = df[df['exer_id:token'].isin(selected_items)].reset_index(drop=True)
        kwargs['df'] = df
        
        kwargs = super().process(**kwargs)

        selected_items = kwargs['df']['exer_id:token'].unique()
        kwargs['df_cpt_relation'] = kwargs['df_cpt_relation'][kwargs['df_cpt_relation']['cpt_head:token'].isin(selected_items)]
        kwargs['df_cpt_relation'] = kwargs['df_cpt_relation'][kwargs['df_cpt_relation']['cpt_tail:token'].isin(selected_items)]
        kwargs['df_cpt_relation'] = kwargs['df_cpt_relation'].reset_index(drop=True)
        return kwargs


run_edustudio(
    dataset='JunyiExerAsCpt',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'EduTrainTPL',   
        'batch_size': 2048,
        'eval_batch_size': 2048,
    },
    datatpl_cfg_dict={
        'cls': 'HierCDFDataTPL',
        # 'load_data_from': 'rawdata',
        # 'raw2mid_op': 'R2M_JunyiExerAsCpt',
        'mid2cache_op_seq': ['M2C_Label2Int', M2C_MyFilterRecords4CD, 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
        'M2C_ReMapId': {
            'share_id_columns': [{'cpt_seq:token_seq', 'cpt_head:token', 'cpt_tail:token', 'exer_id:token'}],
        },
        'M2C_MyFilterRecords4CD': {
            "stu_least_records": 60,
        }
    },
    modeltpl_cfg_dict={
        'cls': 'HierCDF',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL'],
    }
)
