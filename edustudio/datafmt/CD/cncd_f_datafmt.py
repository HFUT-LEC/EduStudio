from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
import pandas as pd
from typing import Dict
import torch
import numpy as np
from torch.utils.data import DataLoader, default_collate


class CNCD_F_DataFmt(CDInterDataFmtExtendsQ):
    def __init__(self, cfg, train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], feat_name2type: Dict[str, str], 
                 df_Q, val_dict: Dict[str, torch.tensor] = None
                 ):
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)

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
    
    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq', 'content:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['Q_mat'] = self.Q_mat
        return dic

    def get_extra_data(self):
        return {
            'content': self.df_Q['content'].to_list()
        }

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        ret_dict = {key: default_collate([d[key] for d in batch]) for key in elem if key not in {'Q_mat'}}
        ret_dict['Q_mat'] = elem['Q_mat']
        return ret_dict
