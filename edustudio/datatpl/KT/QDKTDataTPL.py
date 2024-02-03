import networkx as nx
import numpy as np

from .KTInterExtendsQDataTPL import KTInterExtendsQDataTPL
import torch


class QDKTDataTPL(KTInterExtendsQDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats', 'M2C_RandomDataSplit4KT', 'M2C_GenKCSeq','M2C_GenQMat','M2C_QDKT_OP'],
    }

    def get_extra_data(self, **kwargs):
        return {
            'laplacian_matrix': self.final_kwargs['laplacian_matrix'],
            'train_dict': self.train_dict
        }

    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        self.train_dict = self.dict_train_folds[fold_id]
