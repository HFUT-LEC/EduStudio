from ..common import EduDataTPL


class MGCDDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_MGCD_OP', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    }

    def get_extra_data(self):
        dic = super().get_extra_data()
        dic.update({
            'inter_student': self.final_kwargs['df_inter_stu'],
            'df_G': self.final_kwargs['df_stu']
        })
        return dic

    def df2dict(self):
        super().df2dict()
        self._unwrap_feat(self.final_kwargs['df_inter_stu'])
        self._unwrap_feat(self.final_kwargs['df_stu'])
