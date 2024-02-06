from ..common import EduDataTPL

class FAIRDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilteringRecordsByAttr', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    }

