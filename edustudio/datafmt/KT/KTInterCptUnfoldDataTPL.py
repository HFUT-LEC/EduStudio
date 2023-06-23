from ..common import EduDataTPL
import numpy as np


class KTInterCptUnfoldDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_GenUnFoldCptSeq', 'M2C_BuildSeqInterFeats'],
        'M2C_BuildSeqInterFeats': {
            'seed': 2023,
            'divide_by': 'stu',
            'window_size': -1,
            "divide_scale_list": [7,1,2],
            "extra_inter_feats": ['cpt_unfold:token']
        }
    }

