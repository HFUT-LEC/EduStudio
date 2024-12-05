from ..common import EduDataTPL


class LPKTDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats', 'M2C_RandomDataSplit4KT', 'M2C_LPKT_OP', "M2C_GenQMat"],
        'M2C_BuildSeqInterFeats': {
            "extra_inter_feats": ['start_timestamp:float', 'answer_time:float']
        }
    }

    def set_info_for_fold(self, fold_id):
        dt_info = self.datatpl_cfg['dt_info']
        dt_info['answer_time_count'] = dt_info['answer_time_count_list'][fold_id]
        dt_info['interval_time_count'] = dt_info['interval_time_count_list'][fold_id]
