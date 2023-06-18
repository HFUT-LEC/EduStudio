from ..base_datafmt import BaseDataFmt
import pandas as pd
from typing import Dict, Any
import torch
import os
import numpy as np
from enum import Enum
import copy
from torch.utils.data import DataLoader, default_collate
from ..utils import SpliterUtil

class DataFmtMode(Enum):
    TRAIN=1
    VALID=2
    TEST=3
    MANAGER=4


class CDInterDataFmt(BaseDataFmt):
    default_cfg = {
        'seed': 2023,
        'is_dataset_divided': True,
        'seperator': ',',
        'divide_method': 'same_dist_by_stu',
        'inter_exclude_feat_names': (),
        'divide_scale_list': (7, 1, 2)
    }

    def __init__(self, cfg,
                    train_dict: Dict[str, Any], 
                    test_dict: Dict[str, Any],
                    feat_name2type: Dict[str, str],
                    val_dict: Dict[str, Any] = None,
                    mode: DataFmtMode=DataFmtMode.MANAGER,
                 ):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.val_dict = val_dict
        self.mode = mode
        self.feat_name2type = feat_name2type
        super().__init__(cfg)


    def set_mode(self, mode):
        self.mode = mode
        if self.mode is DataFmtMode.MANAGER:
            self.dict_data = None
        elif self.mode is DataFmtMode.TRAIN:
            self.dict_data = self.train_dict
        elif self.mode is DataFmtMode.VALID:
            self.dict_data = self.val_dict
        elif self.mode is DataFmtMode.TEST:
            self.dict_data = self.test_dict
        else:
            raise ValueError(f"unknown type of mode:{self.mode}")

        self.length = next(iter(self.dict_data.values())).shape[0]

    def _check_params(self):
        super()._check_params()
        assert 'dt_info' not in self.datafmt_cfg
        assert 2 <= len(self.datafmt_cfg['divide_scale_list']) <= 3
        assert np.sum(self.datafmt_cfg['divide_scale_list']) == 10
        assert self.datafmt_cfg.divide_method in ['same_dist_by_stu']

    def _copy(self):
        return copy.copy(self)

    def build_datasets(self):
        train_dataset = self._copy()
        train_dataset.set_mode(DataFmtMode.TRAIN)

        val_dataset = None
        if self.val_dict is not None:
            val_dataset = self._copy()
            val_dataset.set_mode(DataFmtMode.VALID)
        
        test_dataset = self._copy()
        test_dataset.set_mode(DataFmtMode.TEST)
        
        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)

    def build_dataloaders(self):
        batch_size = self.trainfmt_cfg['batch_size']
        num_workers = self.trainfmt_cfg['num_workers']
        eval_batch_size = self.trainfmt_cfg['eval_batch_size']
        train_dataset, val_dataset, test_dataset = self.build_datasets()
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        return train_loader, val_loader, test_loader

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return {
            k: v[index] for k,v in self.dict_data.items()
        }

    def _stat_dataset_info(self):
        self.train_num = self.train_dict['stu_id'].shape[0]
        self.val_num = 0 if self.val_dict is None else self.val_dict['stu_id'].shape[0]
        self.test_num = self.test_dict['stu_id'].shape[0]

        self.stu_count = (torch.cat(
            [self.train_dict['stu_id'], self.test_dict['stu_id']] if self.val_dict is None else 
            [self.train_dict['stu_id'], self.val_dict['stu_id'], self.test_dict['stu_id']]
        ).max()).item() + 1
        self.exer_count = (torch.cat(
            [self.train_dict['exer_id'], self.test_dict['exer_id']] if self.val_dict is None else 
            [self.train_dict['exer_id'], self.val_dict['exer_id'], self.test_dict['exer_id']]
        ).max()).item() + 1

        self.datafmt_cfg['dt_info'].update({
            'stu_count': self.stu_count,
            'exer_count': self.exer_count,
            'trainset_count': self.train_num,
            'valset_count': self.val_num,
            'testset_count': self.test_num
        })
    
    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)
        return cls(
            cfg=cfg,
            train_dict=cls.df2dict(train_df),
            test_dict=cls.df2dict(test_df),
            val_dict=cls.df2dict(val_df) if val_df is not None else None,
            feat_name2type=feat_name2type
        )

    @classmethod
    def read_data(cls, cfg):
        if not os.path.exists(f'{cfg.frame_cfg.data_folder_path}'):
            cls.download_dataset(cfg)
            
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']
        assert len(set(exclude_feats) & {
                   'stu_id:token', 'exer_id:token', 'label:float'}) == 0
        train_df, val_df, test_df = None, None, None
        feat_name2type = {}
        if cfg.datafmt_cfg['is_dataset_divided']:
            train_df, val_df, test_df = cls._read_data_from_divided(cfg)
            train_feat_name2type, train_df = cls._convert_df_to_std_fmt(
                train_df)
            test_feat_name2type, test_df = cls._convert_df_to_std_fmt(test_df)
            feat_name2type.update(train_feat_name2type)
            feat_name2type.update(test_feat_name2type)

            if val_df is not None:
                val_feat_name2type, val_df = cls._convert_df_to_std_fmt(val_df)
                feat_name2type.update(val_feat_name2type)
        else:
            inter_df = cls._read_data_from_undivided(cfg)
            feat_name2type, inter_df = cls._convert_df_to_std_fmt(inter_df)

            if cfg.datafmt_cfg['divide_method'] == 'same_dist_by_stu':
                train_df, val_df, test_df = SpliterUtil.divide_data_df(
                    inter_df, divide_scale_list=cfg.datafmt_cfg['divide_scale_list'],
                    seed=cfg.datafmt_cfg['seed'], label_field='stu_id'
                )
            else:
                raise NotImplementedError

        return feat_name2type, train_df, val_df, test_df

    @classmethod
    def _read_data_from_divided(cls, cfg):
        train_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-train.inter.csv'
        val_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-val.inter.csv'
        test_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-test.inter.csv'

        sep = cfg.datafmt_cfg['seperator']

        train_headers = pd.read_csv(train_file_path, nrows=0).columns.tolist()
        test_headers = pd.read_csv(test_file_path, nrows=0).columns.tolist()
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']

        inter_train_df = pd.read_csv(
            train_file_path, sep=sep, encoding='utf-8', usecols=set(train_headers) - set(exclude_feats)
        )
        inter_val_df = None
        if os.path.exists(val_file_path):
            val_headers = pd.read_csv(val_file_path, nrows=0).columns.tolist()
            inter_val_df = pd.read_csv(
                val_file_path, sep=sep, encoding='utf-8', usecols=set(val_headers) - set(exclude_feats)
            )
        inter_test_df = pd.read_csv(
            test_file_path, sep=sep, encoding='utf-8', usecols=set(test_headers) - set(exclude_feats)
        )

        return inter_train_df, inter_val_df, inter_test_df

    @classmethod
    def _read_data_from_undivided(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}.inter.csv'
        headers = pd.read_csv(file_path, nrows=0).columns.tolist()
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']
        inter_df = pd.read_csv(
            file_path, sep=cfg.datafmt_cfg['seperator'], encoding='utf-8',
            usecols=set(headers) - set(exclude_feats)
        )
        return inter_df

    @staticmethod
    def _convert_df_to_std_fmt(df: pd.DataFrame, inplace=True):
        feat_name2type = {}
        for col in df.columns:
            col_name, col_type = col.split(":")
            feat_name2type[col_name] = col_type
            if col_type == 'token':
                df[col] = df[col].astype('int64')
            elif col_type == 'float':
                df[col] = df[col].astype('float32')
            elif col_type == 'token_seq':
                pass
            elif col_type == 'float_seq':
                pass
            else:
                raise ValueError(f"unknown field type of {col_type}")

            if inplace is True:
                df.rename(columns={col: col_name}, inplace=True)
            else:
                raise NotImplementedError
        return feat_name2type, df

    @staticmethod
    def df2dict(dic):
        return {k: torch.from_numpy(dic[k].to_numpy()) for k in dic}

