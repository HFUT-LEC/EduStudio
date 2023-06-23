from ..common import EduDataTPL
import numpy as np


class KTInterCptUnfoldDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_GenUnFoldCptSeq', 'M2C_BuildSeqInterFeats'],
        'M2C_BuildSeqInterFeats': {
            "extra_inter_feats": ['start_timestamp:float', 'cpt_unfold:token']
        }
    }
