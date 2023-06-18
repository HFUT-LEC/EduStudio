from .inter_datafmt import CDInterDataFmt
import pandas as pd
from typing import Dict
import torch
import numpy as np
from itertools import chain
from torch.utils.data import DataLoader, default_collate


class CDInterDataFmtExtendsQ(CDInterDataFmt):
    def __init__(self, cfg, train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], feat_name2type: Dict[str, str], 
                 df_Q, val_dict: Dict[str, torch.tensor] = None
                 ):
        self.df_Q = df_Q
        super().__init__(cfg, train_dict, test_dict, feat_name2type, val_dict)

    def _init_data_after_dt_info(self):
        super()._init_data_after_dt_info()
        self.Q_mat = self._get_Q_mat_from_df_arr(
            self.df_Q, 
            self.datafmt_cfg['dt_info']['exer_count'], 
            self.datafmt_cfg['dt_info']['cpt_count']
        )
    
    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)
        name2type, df_Q = cls.read_Q_matrix(cfg)
        feat_name2type.update(name2type)

        return cls(
            cfg=cfg,
            train_dict=cls.df2dict(train_df),
            test_dict=cls.df2dict(test_df),
            val_dict=cls.df2dict(val_df) if val_df is not None else None,
            df_Q=df_Q,
            feat_name2type=feat_name2type
        )

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['Q_mat'] = self.Q_mat
        return dic

    def _stat_dataset_info(self):
        super()._stat_dataset_info()
        exer_count = max(self.datafmt_cfg['dt_info']['exer_count'], self.df_Q['exer_id'].max()+1)
        self.datafmt_cfg['dt_info']['exer_count'] = exer_count
        
        cpt_count = len(set(list(chain(*self.df_Q['cpt_seq'].to_list()))))
        self.datafmt_cfg['dt_info']['cpt_count'] = cpt_count
    
    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        ret_dict = {key: default_collate([d[key] for d in batch]) for key in elem if key not in {'Q_mat'}}
        ret_dict['Q_mat'] = elem['Q_mat']
        return ret_dict

    def _get_Q_mat_from_df_arr(self, df_Q_arr, exer_count, cpt_count):
        Q_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
        for _, item in df_Q_arr.iterrows():
            for cpt_id in item['cpt_seq']: Q_mat[item['exer_id'], cpt_id] = 1
        return Q_mat
