import numpy as np

from ..common import EduDataTPL

class CL4KTDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_GenUnFoldKCSeq', 'M2C_CL4KT_OP'],
        'M2C_CL4KT_OP': {
            'sequence_truncation': 'recent',
        }
    }

    def __getitem__(self, index):
        dic = super().__getitem__(index)

        return dic
