from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
import pandas as pd
from typing import Dict
import torch
import numpy as np
from torch.utils.data import DataLoader, default_collate


class CNCD_Q_DataFmt(CDInterDataFmtExtendsQ):
    def __init__(self, cfg, train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], feat_name2type: Dict[str, str], 
                 df_Q, val_dict: Dict[str, torch.tensor] = None
                 ):
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)

    def _init_data_before_dt_info(self):
        super()._init_data_before_dt_info()
        self.Q_mat, self.Q_mask_mat, self.knowledge_pairs = self._get_Q_mat_from_df_arr(
            self.df_Q, self.datafmt_cfg['dt_info']['exer_count'], self.datafmt_cfg['dt_info']['cpt_count']
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
    
    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq', 'cpt_pre_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        df_Q['cpt_pre_seq'] = df_Q['cpt_pre_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['Q_mat'] = self.Q_mat
        dic['knowledge_pairs'] = self.knowledge_pairs
        return dic
    
    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        ret_dict = {key: default_collate([d[key] for d in batch]) for key in elem if key not in {'Q_mat',"knowledge_pairs"}}
        ret_dict['Q_mat'] = elem['Q_mat']
        ret_dict['knowledge_pairs'] = elem['knowledge_pairs']
        return ret_dict

    def _get_Q_mat_from_df_arr(self, df_Q_arr, exer_count, cpt_count):
        Q_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
        Q_mask_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
        knowledge_pairs = []
        kn_tags = []
        kn_topks = []
        for _, item in df_Q_arr.iterrows():
            # kn_tags.append(item['cpt_seq'])
            # kn_topks.append(item['cpt_pre_seq'])
            kn_tags = item['cpt_seq']
            kn_topks = item['cpt_pre_seq']
            knowledge_pairs.append((kn_tags, kn_topks))
            for cpt_id in item['cpt_seq']:
                Q_mat[item['exer_id'], cpt_id-1] = 1
                Q_mask_mat[item['exer_id'], cpt_id-1] = 1
            for cpt_id in item['cpt_pre_seq']:
                Q_mask_mat[item['exer_id'], cpt_id-1] = 1
        return Q_mat, Q_mask_mat, knowledge_pairs
