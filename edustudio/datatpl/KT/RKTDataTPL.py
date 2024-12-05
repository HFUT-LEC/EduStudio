import networkx as nx
import numpy as np

from scipy import sparse
from ..common import EduDataTPL


class RKTDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId','M2C_GenQMat', 'M2C_BuildSeqInterFeats', 'M2C_RandomDataSplit4KT'],
        'M2C_BuildSeqInterFeats': {
            "extra_inter_feats": ['start_timestamp:float']
        }
    }

    def process_load_data_from_middata(self):
        super().process_load_data_from_middata()
        self.final_kwargs['pro_pro_dense'] = self.get_pro_pro_corr()

    def get_pro_pro_corr(self):
        # reference: https://github.com/shalini1194/RKT/issues/2
        pro_cpt_adj = []
        pro_num = self.cfg['datatpl_cfg']['dt_info']['exer_count']
        cpt_num = self.cfg['datatpl_cfg']['dt_info']['cpt_count']
        for index in range(len(self.df_exer)):
            tmp_df = self.df_exer.iloc[index]
            exer_id = tmp_df['exer_id:token']
            cpt_seq = tmp_df['cpt_seq:token_seq']
            for cpt in cpt_seq:
                pro_cpt_adj.append([exer_id, cpt, 1])
        pro_cpt_adj = np.array(pro_cpt_adj).astype(np.int32)
        pro_cpt_sparse = sparse.coo_matrix((pro_cpt_adj[:, 2].astype(np.float32),
                                              (pro_cpt_adj[:, 0], pro_cpt_adj[:, 1])), shape=(pro_num, cpt_num))
        pro_cpt_csc = pro_cpt_sparse.tocsc()
        pro_cpt_csr = pro_cpt_sparse.tocsr()
        pro_pro_adj = []
        for p in range(pro_num):
            tmp_skills = pro_cpt_csr.getrow(p).indices
            similar_pros = pro_cpt_csc[:, tmp_skills].indices
            zipped = zip([p] * similar_pros.shape[0], similar_pros)
            pro_pro_adj += list(zipped)

        pro_pro_adj = list(set(pro_pro_adj))
        pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
        data = np.ones(pro_pro_adj.shape[0]).astype(np.float32)
        pro_pro_sparse = sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))
        return 1-pro_pro_sparse.tocoo().toarray()

    def get_extra_data(self):
        return {
            "pro_pro_dense": self.final_kwargs['pro_pro_dense']
        }

    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        self.train_dict = self.dict_train_folds[fold_id]