from ..common import EduDataTPL
import torch
from typing import Dict
import pandas as pd
import os

class HierCDFDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
        'M2C_ReMapId': {
            'share_id_columns': [{'cpt_seq:token_seq', 'cpt_head:token', 'cpt_tail:token'}],
        }
    }
    
    def __init__(self, cfg, df_cpt_relation=None, **kwargs):
        self.df_cpt_relation = df_cpt_relation
        super().__init__(cfg, **kwargs)

    @property
    def common_str2df(self):
        return {
            "df": self.df, "df_train": self.df_train, "df_valid": self.df_valid,
            "df_test": self.df_test, "dt_info": self.datatpl_cfg['dt_info'], 
            "df_stu": self.df_stu, "df_exer": self.df_exer,
            "df_cpt_relation": self.df_cpt_relation
        }
    
    @classmethod
    def load_data(cls, cfg): # 只在middata存在时调用
        kwargs = super().load_data(cfg)
        new_kwargs = cls.load_data_from_cpt_relation(cfg)
        for df in new_kwargs.values(): 
            if df is not None:
                cls._preprocess_feat(df) # 类型转换
        kwargs.update(new_kwargs)
        return kwargs

    def get_extra_data(self):
        extra_dict = super().get_extra_data()
        extra_dict['cpt_relation_edges'] = self._unwrap_feat(self.final_kwargs['df_cpt_relation'])
        return extra_dict
        
    @classmethod
    def load_data_from_cpt_relation(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.cpt_relation.prerequisite.csv'
        df_cpt_relation = None
        if os.path.exists(file_path):
            sep = cfg.datatpl_cfg['seperator']
            df_cpt_relation = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['cpt_head:token', 'cpt_tail:token'])
        return {"df_cpt_relation": df_cpt_relation}
