from ..common import EduDataTPL

class FAIRCDDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    }

    def get_extra_data(self, **kwargs):
        extra_data = super().get_extra_data(**kwargs)
        df_stu_dict = {
            'df_stu': self.df_stu
        }
        extra_data.update(df_stu_dict)
        return extra_data
