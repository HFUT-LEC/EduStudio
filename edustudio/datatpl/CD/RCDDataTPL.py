from ..common import EduDataTPL
import json
import numpy as np


class RCDDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': [
            'M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 
            'M2C_RandomDataSplit4CD', 'M2C_BuildKCRelation', 
            'M2C_GenQMat', 'M2C_RCD_OP'
        ],
        'M2C_BuildKCRelation': {
            'relation_type': 'rcd_transition',
            'threshold': None
        }
    }

    def get_extra_data(self):
        extra_dict = super().get_extra_data()
        extra_dict['local_map'] = self.local_map
        return extra_dict

    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        self.local_map = self.final_kwargs['local_map']
        self.local_map['u_from_e'] = self.local_map['u_from_e_list'][fold_id]
        self.local_map['e_from_u'] = self.local_map['e_from_u_list'][fold_id]
