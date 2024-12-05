from ..common import EduDataTPL

class CNCDFDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    }

    def get_extra_data(self):
        return {
            'content': self.df_exer['content:token_seq'].to_list(),
            'Q_mat': self.final_kwargs['Q_mat']
        }
