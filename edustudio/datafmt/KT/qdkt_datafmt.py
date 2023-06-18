from .kt_inter_datafmt_extends_q import KTInterDataFmtExtendsQ
import numpy as np
from ..utils import PadSeqUtil
import pandas as pd
import torch
from collections import defaultdict
import math
import networkx as nx


class QDKTDataFmt(KTInterDataFmtExtendsQ):
    default_cfg = {

    }
    def __init__(self, cfg, train_dict, val_dict, test_dict, feat2type, **kwargs):
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type, **kwargs)

    def get_extra_data(self):
        graph = self.generate_graph()
        laplacian_matrix = self.laplacian_matrix(graph)
        return {
            'laplacian_matrix': laplacian_matrix,
            'train_dict': self.train_dict
        }

    def generate_graph(self):
        graph = nx.Graph()
        len1 = len(self.Q_mat)
        graph.add_nodes_from([i for i in range(1, len1 + 1)])
        for index in range(len1-1):
            for bindex in range(index + 1, len1):
                if not (False in (self.Q_mat[index, :] == self.Q_mat[bindex, :]).tolist()):
                    graph.add_edge(index + 1, bindex + 1)
        return graph


 # 求图的拉普拉斯矩阵 L = D - A
    def laplacian_matrix(self,graph):
        # 求邻接矩阵
        A = np.array(nx.adjacency_matrix(graph).todense())
        A = -A
        for i in range(len(A)):
            # 求顶点的度
            degree_i = graph.degree(i+1)  # 节点编号从1开始，若从0开始，将i+1改为i
            A[i][i] = A[i][i] + degree_i
        return A


