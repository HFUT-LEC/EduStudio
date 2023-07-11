from ..common import BaseMid2Cache
import networkx as nx
import numpy as np
from itertools import combinations, chain
import heapq
from collections import defaultdict


class M2C_CDGK_OP(BaseMid2Cache):
    default_cfg = {
        'subgraph_count': 1, 
    }

    def process(self, **kwargs):
        cpt_count = kwargs['dt_info']['cpt_count']
        exer_count = kwargs['dt_info']['exer_count']
        df_exer = kwargs['df_exer']
        subgraph_count = self.m2c_cfg['subgraph_count']

        if subgraph_count == 1:
            kwargs['n_group_of_cpt'] = np.ones(shape=(cpt_count, ), dtype=np.int64)
            kwargs['gid2exers'] = {0: np.arange(exer_count)}
            kwargs['dt_info']['n_cpt_group'] = 1
            return kwargs

        G = self.get_G_from_cpt_concurrence(df_exer, cpt_count, exer_count)
        number_connected_components = nx.number_connected_components(G)

        cpt_group_list = None
        if subgraph_count == number_connected_components or subgraph_count < 0:
            cpt_group_list = nx.connected_components(G)
        elif subgraph_count < number_connected_components:
            # 小根堆，合并，记录合并索引
            subgraphs = list(nx.connected_components(G))
            cpt_group_list = self._merge_nodes_list(subgraphs, subgraph_count)
        else:
            raise NotImplementedError

        cpt2groups = defaultdict(list)
        for gid, conn_comp in enumerate(cpt_group_list):
            for cpt_id in conn_comp: cpt2groups[cpt_id].append(gid)
   
        kwargs['n_group_of_cpt'] = np.array([len(v) for v in cpt2groups.values()], dtype=np.int64)
        kwargs['gid2exers'] = defaultdict(set)
        for _, tmp_df in df_exer[['exer_id:token', 'cpt_seq:token_seq']].iterrows():
            exer_id = tmp_df['exer_id:token']
            cpt_seq = tmp_df['cpt_seq:token_seq']

            for cpt_id in cpt_seq:
                for gid in cpt2groups[cpt_id]:
                    kwargs['gid2exers'][gid].add(exer_id)
        kwargs['gid2exers'] = {gid: np.array(list(exers)) for gid, exers in kwargs['gid2exers'].items()}
        kwargs['dt_info']['n_cpt_group'] = len(kwargs['gid2exers'])

        return kwargs
    

    @staticmethod
    def get_G_from_cpt_concurrence(df_exer, cpt_count, exer_count):
        G = nx.Graph()
        G.add_nodes_from(np.arange(cpt_count))
        df_exer = df_exer.set_index('exer_id:token')
        for exer_id in range(exer_count):
            for edge in combinations(df_exer.loc[exer_id].tolist()[0], 2):
                G.add_edge(edge[0], edge[1])
        return G

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
