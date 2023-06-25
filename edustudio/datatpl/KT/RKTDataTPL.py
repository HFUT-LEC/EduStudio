import networkx as nx
import numpy as np

from scipy import sparse
from .KTInterDataTPL import KTInterDataTPL


class RKTDataTPL(KTInterDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats'],
        'M2C_BuildSeqInterFeats': {
            "extra_inter_feats": ['start_timestamp:float']
        }
    }


    def get_corr_data(self):
        file_path = f'{self.cfg.frame_cfg.data_folder_path}/pro_pro_sparse.npz'
        # pro_pro_dense = np.zeros((self.n_item, self.n_item))
        pro_pro_sparse = sparse.load_npz(file_path)
        pro_pro_coo = pro_pro_sparse.tocoo()
        # print(pro_skill_csr)
        self.pro_pro_dense = pro_pro_coo.toarray()
        return self.pro_pro_dense


    def get_extra_data(self):
        return {
            "pro_pro_dense": self.get_corr_data()
        }
