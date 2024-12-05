from ..common import EduDataTPL


class DKTForgetDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats','M2C_RandomDataSplit4KT', "M2C_DKTForget_OP"],
        'M2C_BuildSeqInterFeats': {
            "extra_inter_feats": ['start_timestamp:float']
        }
    }

    def set_info_for_fold(self, fold_id):
        dt_info = self.datatpl_cfg['dt_info']
        dt_info['n_pcount'] = dt_info['n_pcount_list'][fold_id]
        dt_info['n_rgap'] = dt_info['n_rgap_list'][fold_id]
        dt_info['n_sgap'] = dt_info['n_sgap_list'][fold_id]
