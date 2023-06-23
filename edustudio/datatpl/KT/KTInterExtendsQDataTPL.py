from ..common import EduDataTPL
import numpy as np


class KTInterExtendsQDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats', 'M2C_GenCptSeq'],
    }

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['cpt_seq'] = np.stack(
            [self.cpt_seq_padding[exer_seq] for exer_seq in dic['exer_seq']], axis=0
        )
        dic['cpt_seq_mask'] = np.stack(
            [self.cpt_seq_mask[exer_seq] for exer_seq in dic['exer_seq']], axis=0
        )
        return dic
