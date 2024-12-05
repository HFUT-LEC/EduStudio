import pandas as pd
from .general_datatpl import DataTPLStatus
from edustudio.utils.common import UnifyConfig
from .general_datatpl import GeneralDataTPL
import os
import torch


class EduDataTPL(GeneralDataTPL):
    """Educational Data Template including student features and exercise features
    """
    
    default_cfg = {
        'exer_exclude_feat_names': (),
        'stu_exclude_feat_names': (),
    }

    def __init__(
                self, cfg:UnifyConfig,
                df: pd.DataFrame=None,
                df_train: pd.DataFrame=None,
                df_valid: pd.DataFrame=None,
                df_test: pd.DataFrame=None,
                df_stu: pd.DataFrame=None,
                df_exer: pd.DataFrame=None,
                status: DataTPLStatus=DataTPLStatus()
            ):
        self.df_stu = df_stu
        self.df_exer = df_exer
        super().__init__(cfg, df, df_train, df_valid, df_test, status)
        assert self.df_stu is not None or self.df_exer is not None

    @property
    def common_str2df(self):
        return {
            "df": self.df, "df_train": self.df_train, "df_valid": self.df_valid,
            "df_test": self.df_test, "df_stu": self.df_stu, "df_exer": self.df_exer, "dt_info": self.datatpl_cfg['dt_info']
        }

    @classmethod
    def load_data(cls, cfg): # 只在middata存在时调用
        kwargs = super().load_data(cfg)
        new_kwargs = cls.load_data_from_side_information(cfg)
        for df in new_kwargs.values(): 
            if df is not None:
                cls._preprocess_feat(df) # 类型转换
        kwargs.update(new_kwargs)
        return kwargs
    
    @classmethod
    def load_data_from_side_information(cls, cfg):
        stu_fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.stu.csv'
        exer_fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.exer.csv'

        sep = cfg.datatpl_cfg['seperator']
        stu_exclude_feats = cfg.datatpl_cfg['stu_exclude_feat_names']
        exer_exclude_feats = cfg.datatpl_cfg['exer_exclude_feat_names']
        assert len(set(stu_exclude_feats) & {'stu_id:token'}) == 0
        assert len(set(exer_exclude_feats) & {'exer_id:token'}) == 0

        df_stu, df_exer = None, None
        if os.path.exists(stu_fph):
            df_stu = cls._load_atomic_csv(stu_fph, exclude_headers=stu_exclude_feats, sep=sep)

        if os.path.exists(exer_fph):
            df_exer = cls._load_atomic_csv(exer_fph, exclude_headers=exer_exclude_feats, sep=sep)

        return {"df_stu": df_stu, "df_exer": df_exer}
    
    @property
    def hasStuFeats(self):
        return self.df_stu is not None
    
    @property
    def hasExerFeats(self):
        return self.df_exer is not None

    @property
    def hasQmat(self):
        return self.hasExerFeats and "cpt_list" in self.df_exer.columns

    def save_cache(self):
        super().save_cache()
        df_stu_fph = f"{self.cache_folder_path}/df_stu.pkl"
        df_exer_fph = f"{self.cache_folder_path}/df_exer.pkl"
        self.save_pickle(df_stu_fph, self.df_stu)
        self.save_pickle(df_exer_fph, self.df_exer)
    
    def load_cache(self):
        super().load_cache()
        df_stu_fph = f"{self.cache_folder_path}/df_stu.pkl"
        df_exer_fph = f"{self.cache_folder_path}/df_exer.pkl"
        self.df_stu = self.load_pickle(df_stu_fph)
        self.df_exer = self.load_pickle(df_exer_fph)
