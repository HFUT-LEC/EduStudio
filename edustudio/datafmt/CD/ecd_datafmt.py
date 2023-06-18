from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
import pandas as pd
from typing import Dict
import torch
import numpy as np
from torch.utils.data import DataLoader, default_collate


class ECD_DataFmt(CDInterDataFmtExtendsQ):
    default_cfg = {
        'seed': 2023,
        'is_dataset_divided': False,
    }
    def __init__(self, cfg, train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], feat_name2type: Dict[str, str], 
                 df_Q, df_QQQ, group_list, val_dict: Dict[str, torch.tensor] = None
                 ):
        self.QQQ_list = df_QQQ['qqq_seq'].to_list()
        qqq_list = []
        for v in self.QQQ_list:
            ls = [int(i) for i in v.split(',')]
            qqq_list.append(ls)
        self.QQQ_list = torch.tensor(qqq_list)
        self.group_list = group_list
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)

    def _stat_dataset_info(self):
        super()._stat_dataset_info()
        qqq_count =  torch.max(self.QQQ_list) + 1
        self.datafmt_cfg['dt_info']['qqq_count'] = qqq_count

    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)
        name2type, df_Q = cls.read_Q_matrix(cfg)
        _, df_QQQ = cls.read_QQQ_csv(cfg)
        _, group_list = cls.read_QQQ_group(cfg)
        feat_name2type.update(name2type)

        return cls(
            cfg=cfg,
            train_dict=cls.df2dict(train_df),
            test_dict=cls.df2dict(test_df),
            val_dict=cls.df2dict(val_df) if val_df is not None else None,
            df_Q=df_Q,
            df_QQQ = df_QQQ,
            group_list = group_list,
            feat_name2type=feat_name2type
        )
    
    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q

    @classmethod
    def read_QQQ_csv(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-QQQ.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_QQQ = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['stu_id:token', 'qqq_seq:token_seq'])
        feat_name2type, df_QQQ = cls._convert_df_to_std_fmt(df_QQQ)
        # df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_QQQ

    @classmethod
    def read_QQQ_group(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}_QQQ-group.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_QQQ_group = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['qqq_id:token', 'group_id:token'])
        feat_name2type, df_QQQ_group = cls._convert_df_to_std_fmt(df_QQQ_group)
        gp = df_QQQ_group.groupby("group_id")['qqq_id']
        gps = gp.groups
        gps_list = []
        for k, v in gps.items():
            #     print(list(v))
            gps_list.append(list(v))
        return feat_name2type, gps_list

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['Q_mat'] = self.Q_mat
        dic['QQQ_list'] = self.QQQ_list
        return dic

    def get_extra_data(self):
        return {
            'qqq_group_list': self.group_list
        }

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        ret_dict = {key: default_collate([d[key] for d in batch]) for key in elem if key not in {'Q_mat','QQQ_list'}}
        ret_dict['Q_mat'] = elem['Q_mat']
        ret_dict['QQQ_list'] = elem['QQQ_list']
        return ret_dict
