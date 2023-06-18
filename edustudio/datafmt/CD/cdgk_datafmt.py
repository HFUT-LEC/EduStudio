from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
import pandas as pd
import networkx as nx
import torch
from typing import Dict
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, default_collate
from itertools import combinations
import heapq


class CDGKDataFmt(CDInterDataFmtExtendsQ):
    default_cfg = {
        'have_cpt2group': False,
        'subgraph_count': 2
    }
    def __init__(self, cfg, 
                 train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], 
                 feat_name2type: Dict[str, str], cpt2group,
                 df_Q, val_dict: Dict[str, torch.tensor] = None
        ):
        self.cpt2group = cpt2group
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)

        if self.cpt2group is None:
            if self.datafmt_cfg['have_cpt2group']:
                raise ValueError("no cpt2group")
            else:
                self.construct_graph()

    def _init_data_before_dt_info(self):
        super()._init_data_before_dt_info()
        self.inter2group = defaultdict(list)
        train_count = next(iter(self.train_dict.values())).shape[0]
        val_count = next(iter(self.val_dict.values())).shape[0] if self.val_dict is not None else 0
        test_count = next(iter(self.test_dict.values())).shape[0]
        # idx = 0
        # df_Q = self.df_Q.set_index('exer_id')
        # for i in range(train_count):
        #     exer_id = self.train_dict['exer_id'][i].item()
        #     for cpt_id in df_Q.loc[exer_id]:
        #         self.inter2group[idx].append(self.cpt2group[cpt_id])
        #     idx+=1
        # if self.val_dict:
        #     for i in range(val_count):
        #         exer_id = self.val_dict['exer_id'][i].item()
        #         for cpt_id in df_Q.loc[exer_id]:
        #             self.inter2group[idx].append(self.cpt2group[cpt_id])
        #         idx+=1
        # for i in range(test_count):
        #     exer_id = self.test_dict['exer_id'][i].item()
        #     for cpt_id in df_Q.loc[exer_id]:
        #         self.inter2group[idx].append(self.cpt2group[cpt_id])
        #     idx+=1
        # self.cpt2group_mat = self.padding(self.inter2group)
    
    @staticmethod
    def padding(inter2group, padding_value=-1):
        max_len = np.max([len(i) for i in inter2group.values()])
        ret_mat = torch.full((len(inter2group), max_len), fill_value=padding_value, dtype=torch.long)
        for k, v in inter2group.items():
            ret_mat[k] = torch.LongTensor(v + [padding_value] * (max_len - len(v)))
        return ret_mat
        
    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)
        name2type, df_Q = cls.read_Q_matrix(cfg)
        feat_name2type.update(name2type)

        cpt2group = None
        if cfg.datafmt_cfg['have_cpt2group']:
            cpt2group = cls.read_cpt2group(cfg)
        return cls(
            cfg=cfg,
            train_dict=cls.df2dict(train_df),
            test_dict=cls.df2dict(test_df),
            val_dict=cls.df2dict(val_df) if val_df is not None else None,
            df_Q=df_Q, cpt2group=cpt2group,
            feat_name2type=feat_name2type
        )
    
    @classmethod
    def read_cpt2group(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-cpt2group.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_cpt2group = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['cpt_id:token', 'group_id:token'])
        _, df_cpt2group = cls._convert_df_to_std_fmt(df_cpt2group)
        return df_cpt2group.groupby('cpt_id').to_dict()


    def construct_graph(self):
        # 基于训练集构造知识点关系图
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.datafmt_cfg['dt_info']['cpt_count']))
        df_Q = self.df_Q.set_index('exer_id')
        for exer_id in range(self.datafmt_cfg['dt_info']['exer_count']):
            for edge in combinations(df_Q.loc[exer_id].tolist()[0], 2):
                G.add_edge(edge[0], edge[1])
        number_connected_components = nx.number_connected_components(G)
        if self.datafmt_cfg['subgraph_count'] == 1 or self.datafmt_cfg['subgraph_count'] == number_connected_components:
            pass
        elif self.datafmt_cfg['subgraph_count'] < number_connected_components:
            # 小根堆，合并，记录合并索引
            subgraphs = list(nx.connected_components(G))
            subgraphs = self._merge_nodes_list(subgraphs, self.datafmt_cfg['subgraph_count'])
        else:
            # 大根堆，拆分, 最小割集
            pass
    @staticmethod
    def _merge_nodes_list(nodes_list, graph_num):
        """
        合并节点集合，每次选择大小最小的两个集合进行合并

        参数:
        nodes_list (list): 输入的节点集合列表，列表内每一个元素也是一个节点集合
        graph_num (int): 指定返回的列表长度

        返回:
        list: 合并后的节点集合列表
        """
        # 检查输入参数
        if graph_num <= 0 or graph_num > len(nodes_list):
            raise ValueError("graph_num 必须大于0且不超过 nodes_list 的长度")

        # 使用小根堆保存节点集合及其大小
        heap = [(len(nodes), nodes) for nodes in nodes_list]
        heapq.heapify(heap)

        while len(heap) > graph_num:
            # 选择大小最小的两个节点集合进行合并
            len1, nodes1 = heapq.heappop(heap)
            len2, nodes2 = heapq.heappop(heap)
            merged_nodes = nodes1 | nodes2  # 使用集合的并运算合并集合
            heapq.heappush(heap, (len(merged_nodes), merged_nodes))

        # 将合并后的节点集合从小根堆中取出
        merged_nodes_list = [heapq.heappop(heap)[1] for _ in range(len(heap))]

        return merged_nodes_list
        

    def split_subgraph(self):
        self.construct_graph()

    @staticmethod
    def collate_fn(batch):
        pass
    
    def __getitem__(self, index):
        dic = super().__getitem__(index)
        new_idx = index
        if index + 1 > self.train_num + self.val_num:
            new_idx = index - self.train_num - self.val_num
        elif index + 1 > self.train_num:
            new_idx = index - self.train_num
        return {gid: dic for gid in self.inter2group[new_idx]}
    
