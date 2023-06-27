import networkx as nx
from ..common import BaseMid2Cache
import numpy as np


class M2C_QDKT_OP(BaseMid2Cache):
    default_cfg = {}

    def process(self, **kwargs):
        self.df_Q = kwargs['df_exer']
        dt_info = kwargs['dt_info']
        self.num_q = dt_info['exer_count']
        self.num_c = dt_info['cpt_count']
        self.Q_mat = kwargs['Q_mat']
        graph = self.generate_graph()
        laplacian_matrix = self.laplacian_matrix(graph)
        kwargs['laplacian_matrix'] = laplacian_matrix
        return kwargs

    def generate_graph(self):

        graph = nx.Graph()
        len1 = len(self.Q_mat)
        graph.add_nodes_from([i for i in range(1, len1 + 1)])
        for index in range(len1 - 1):
            for bindex in range(index + 1, len1):
                if not (False in (self.Q_mat[index, :] == self.Q_mat[bindex, :]).tolist()):
                    graph.add_edge(index + 1, bindex + 1)
        return graph

        # 求图的拉普拉斯矩阵 L = D - A

    def laplacian_matrix(self, graph):
        # 求邻接矩阵
        A = np.array(nx.adjacency_matrix(graph).todense())
        A = -A
        for i in range(len(A)):
            # 求顶点的度
            degree_i = graph.degree(i + 1)  # 节点编号从1开始，若从0开始，将i+1改为i
            A[i][i] = A[i][i] + degree_i
        return A
