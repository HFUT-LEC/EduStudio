import numpy as np

from .KTInterExtendsQDataTPL import KTInterExtendsQDataTPL
import torch


class DIMKTDataTPL(KTInterExtendsQDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_GenUnFoldKCSeq', 'M2C_BuildSeqInterFeats', 'M2C_RandomDataSplit4KT', 'M2C_GenKCSeq', "M2C_DIMKT_OP"],
        'M2C_BuildSeqInterFeats': {
            # 'window_size': 200,
            "extra_inter_feats": ['start_timestamp:float', 'cpt_unfold:token']
        }
    }

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['qd_seq'] = np.stack(
            [self.q_dif[exer_seq][0] for exer_seq in dic['exer_seq']], axis=0
        )
        dic['cd_seq'] = np.stack(
            [self.c_dif[cpt_seq][0] for cpt_seq in dic['cpt_unfold_seq']], axis=0
        )
        dic['cd_seq'] = np.squeeze(dic['cd_seq'])
        mask = dic['mask_seq']==0
        dic['qd_seq'][mask]=0
        dic['cd_seq'][mask] = 0
        return dic

    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        self.q_dif = self.final_kwargs['q_diff_list'][fold_id]
        self.c_dif = self.final_kwargs['c_diff_list'][fold_id]
