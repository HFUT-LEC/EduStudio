from ..common import EduDataTPL

class KTInterCptAsExerDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ["M2C_CptAsExer", 'M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats'],
    }

