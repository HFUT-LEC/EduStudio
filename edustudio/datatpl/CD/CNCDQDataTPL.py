from ..common import EduDataTPL
import torch
import pandas as pd
import os

class CNCDQDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    }

    @property
    def common_str2df(self):
        return {
            "df": self.df, "df_train": self.df_train, "df_valid": self.df_valid,
            "df_test": self.df_test, "dt_info": self.datatpl_cfg['dt_info'], 
            "df_stu": self.df_stu, "df_exer": self.df_exer,
            "df_questionnaire": self.df_questionnaire
        }

    @classmethod
    def load_data(cls, cfg): # 只在middata存在时调用
        kwargs = super().load_data(cfg)
        new_kwargs = cls.load_data_from_questionnaire(cfg)
        for df in new_kwargs.values(): 
            if df is not None:
                cls._preprocess_feat(df) # 类型转换
        kwargs.update(new_kwargs)
        return kwargs

    @classmethod
    def load_data_from_questionnaire(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.questionnaire.csv'
        df_questionnaire = None
        if os.path.exists(file_path):
            sep = cfg.datatpl_cfg['seperator']
            df_questionnaire = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['cpt_head:token', 'cpt_tail:token'])
        return {"df_questionnaire": df_questionnaire}
