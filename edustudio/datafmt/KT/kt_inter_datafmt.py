from ..base_datafmt import BaseDataFmt
import pandas as pd
from typing import Dict, Any
import torch
import os
import numpy as np
from enum import Enum
import copy
from torch.utils.data import DataLoader, default_collate
from ..utils import PadSeqUtil, SpliterUtil


class DataFmtMode(Enum):
    TRAIN=1
    VALID=2
    TEST=3
    MANAGER=4


class KTInterDataFmt(BaseDataFmt):
    default_cfg = {
        'seed': 2023,
        'is_dataset_divided': True,
        'seperator': ',',
        'inter_exclude_feat_names': (),
        'window_size': 50,
        'divide_scale_list': (7, 1, 2),
        'divide_method': 'by_stu', # ['by_stu', 'by_time']
        'inter_data_type': 'std' # ['std', 'dkt-type'] 
    }

    def __init__(self, cfg,
                 train_dict,
                 val_dict,
                 test_dict,
                 feat2type,
                 mode: DataFmtMode=DataFmtMode.MANAGER,
                 ):
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        self.feat2type = feat2type
        self.mode = mode
        super().__init__(cfg)
        self.transform_data_type()

    def _check_params(self):
        super()._check_params()
        assert 'dt_info' not in self.datafmt_cfg
        assert 2 <= len(self.datafmt_cfg['divide_scale_list']) <= 3
        assert np.sum(self.datafmt_cfg['divide_scale_list']) == 10

    def _copy(self):
        return copy.copy(self)
    
    def transform_data_type(self):
        inter_data_type = self.datafmt_cfg['inter_data_type']
        if inter_data_type == 'std':
            pass
        elif inter_data_type == 'dkt-type':
            self.train_dict['dkt-type-feat'] = self._std2dkt_type(self.train_dict)
            if self.val_dict:
                self.val_dict['dkt-type-feat'] = self._std2dkt_type(self.val_dict)
            self.test_dict['dkt-type-feat'] = self._std2dkt_type(self.test_dict)
        else:
            raise NotImplementedError

    def _std2dkt_type(self, data_dict):
        new_feat_list = []
        n_exers = self.datafmt_cfg['dt_info']['exer_count']
        for i in range(data_dict['exer_seq'].shape[0]):
            exer_seq = data_dict['exer_seq'][i]
            label_seq = data_dict['label_seq'][i]
            mask_seq = data_dict['mask_seq'][i]
            onehot = torch.zeros(size=[len(exer_seq), 2 * n_exers], dtype=torch.int8)
            for j, (exer_id, label, mask) in enumerate(zip(exer_seq, label_seq, mask_seq)):
                if mask:          
                    index = int(exer_id.item() if label.item() > 0 else exer_id.item() + n_exers)
                    onehot[j][index] = 1
                else:
                    pass
            new_feat_list.append(onehot)
        return torch.stack(new_feat_list, dim=0)

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
            [self.train_dict['exer_seq'], self.test_dict['exer_seq']] if self.val_dict is None else 
            [self.train_dict['exer_seq'], self.val_dict['exer_seq'], self.test_dict['exer_seq']]
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
        feat2type, train_dict, val_dict, test_dict = cls.read_data(cfg)
        return cls(
            cfg=cfg,
            train_dict=cls.dict2tensor(train_dict),
            test_dict=cls.dict2tensor(test_dict),
            val_dict=cls.dict2tensor(val_dict) if val_dict is not None else None,
            feat2type=feat2type
        )

    @classmethod
    def read_data(cls, cfg):
        if not os.path.exists(f'{cfg.frame_cfg.data_folder_path}'):
            cls.download_dataset(cfg)
            
        if cfg.datafmt_cfg['is_dataset_divided']:
            feat2type, train_df, val_df, test_df = cls._read_data_from_divided(cfg)
            cfg_window_size = cfg.datafmt_cfg['window_size']
            if cfg_window_size <= 0 or cfg_window_size is None:
                cfg_window_size = np.max([
                    train_df[['stu_id', 'exer_id']].groupby('stu_id').agg('count')['exer_id'].max(),
                    val_df[['stu_id', 'exer_id']].groupby('stu_id').agg('count')['exer_id'].max() if val_df else 0,
                    test_df[['stu_id', 'exer_id']].groupby('stu_id').agg('count')['exer_id'].max()
                ])
                cfg.logger.info(f"actual window size: {cfg_window_size}")
            train_dict = cls.construct_df2dict(cfg, train_df, maxlen=cfg_window_size)
            val_dict = cls.construct_df2dict(cfg, val_df, maxlen=cfg_window_size)
            test_dict = cls.construct_df2dict(cfg, test_df, maxlen=cfg_window_size)
        else:

            feat2type, data_df = cls._read_data_from_undivided(cfg)
            train_dict, val_dict, test_dict = cls._divide_data_df(cfg, data_df)
        
        return feat2type, train_dict, val_dict, test_dict

    @classmethod
    def _read_data_from_divided(cls, cfg):
        # config
        train_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-train.inter.csv'
        val_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-val.inter.csv'
        test_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-test.inter.csv'
        sep = cfg.datafmt_cfg['seperator']
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']

        # read data
        train_headers = pd.read_csv(train_file_path, nrows=0).columns.tolist()
        test_headers = pd.read_csv(test_file_path, nrows=0).columns.tolist()
        inter_train_df = pd.read_csv(
            train_file_path, sep=sep, encoding='utf-8', usecols=set(train_headers) - set(exclude_feats)
        )
        inter_test_df = pd.read_csv(
            test_file_path, sep=sep, encoding='utf-8', usecols=set(test_headers) - set(exclude_feats)
        )
        inter_val_df = None
        if os.path.exists(val_file_path):
            val_headers = pd.read_csv(val_file_path, nrows=0).columns.tolist()
            inter_val_df = pd.read_csv(
                val_file_path, sep=sep, encoding='utf-8', usecols=set(val_headers) - set(exclude_feats)
            )

        # to standard fmt
        d1, inter_train_df = cls._convert_df_to_std_fmt(inter_train_df)
        d2, inter_val_df = cls._convert_df_to_std_fmt(inter_val_df)
        d3, inter_test_df = cls._convert_df_to_std_fmt(inter_test_df)
        feat2type = d1
        feat2type.update(d2)
        feat2type.update(d3)

        # sort
        if 'timestamp' in feat2type:
            inter_train_df = inter_train_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
            if inter_val_df is not None:
                inter_val_df = inter_val_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
            inter_test_df = inter_test_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

        return feat2type, inter_train_df, inter_val_df, inter_test_df

    @classmethod
    def _read_data_from_undivided(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}.inter.csv'
        headers = pd.read_csv(file_path, nrows=0).columns.tolist()
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']
        inter_df = pd.read_csv(
            file_path, sep=cfg.datafmt_cfg['seperator'], encoding='utf-8',
            usecols=set(headers) - set(exclude_feats)
        )
        feat2type, inter_df = cls._convert_df_to_std_fmt(inter_df)
        if 'timestamp' in feat2type:
            inter_df = inter_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        return feat2type, inter_df

    @staticmethod
    def _convert_df_to_std_fmt(df: pd.DataFrame, inplace=True):
        if df is None:
            return {}, df
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

    @classmethod
    def construct_df2dict(cls, cfg, df: pd.DataFrame, maxlen=None):
        if df is None: return None

        tmp_df = df[['stu_id','exer_id','label']].groupby('stu_id').agg(
            lambda x: list(x) 
        ).reset_index()

        exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(
            tmp_df['exer_id'].to_list(), return_idx=True, return_mask=True, 
            maxlen=maxlen
        )
        label_seq, _, _ = PadSeqUtil.pad_sequence(
            tmp_df['label'].to_list(), dtype=np.float32,
            maxlen=maxlen
        )

        stu_id = tmp_df['stu_id'].to_numpy()[idx]
        
        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'label_seq': label_seq,
            'mask_seq': mask_seq
        }

    @classmethod
    def _divide_data_df(cls, cfg, df: pd.DataFrame):
        divide_method = cfg['datafmt_cfg']['divide_method']
        if divide_method == 'by_stu':
            return cls._divide_data_df_by_stu(cfg, df)
        elif divide_method == 'by_time':
            return cls._divide_data_df_by_time(cfg, df)
        else:
            raise ValueError(f"unknown divide method: {divide_method}")
    
    @classmethod
    def _divide_data_df_by_stu(cls, cfg, df: pd.DataFrame):
        
        train_stu_id, val_stu_id, test_stu_id = SpliterUtil.divide_data_df(
            df['stu_id'].drop_duplicates(), seed=cfg.datafmt_cfg['seed'], shuffle=True,
            divide_scale_list=cfg.datafmt_cfg['divide_scale_list']
        )
        train_df = df[df['stu_id'].isin(train_stu_id)]
        val_df = df[df['stu_id'].isin(val_stu_id)] if val_stu_id is not None else None
        test_df = df[df['stu_id'].isin(test_stu_id)]
        cfg_window_size = cfg.datafmt_cfg['window_size']
        if cfg_window_size <= 0 or cfg_window_size is None:
            cfg_window_size = np.max([
                train_df[['stu_id', 'exer_id']].groupby('stu_id').agg('count')['exer_id'].max(),
                val_df[['stu_id', 'exer_id']].groupby('stu_id').agg('count')['exer_id'].max() if val_df else 0,
                test_df[['stu_id', 'exer_id']].groupby('stu_id').agg('count')['exer_id'].max()
            ])
            cfg.logger.info(f"actual window size: {cfg_window_size}")
        train_dict = cls.construct_df2dict(cfg, train_df, maxlen=cfg_window_size)
        val_dict = cls.construct_df2dict(cfg, val_df, maxlen=cfg_window_size)
        test_dict = cls.construct_df2dict(cfg, test_df, maxlen=cfg_window_size)
        return train_dict, val_dict, test_dict

    @classmethod
    def _divide_data_df_by_time(cls, cfg, df: pd.DataFrame):
        data_dict = cls.construct_df2dict(df, maxlen=cfg.datafmt_cfg['window_size'])
        train_dict, val_dict, test_dict = SpliterUtil.divide_data_dict(
            data_dict, seed=cfg.datafmt_cfg['seed'], shuffle=True,
            divide_scale_list=cfg.datafmt_cfg['divide_scale_list'], label_field='stu_id'
        ) # 存在信息泄露？
        return train_dict, val_dict, test_dict

    @staticmethod
    def dict2tensor(dic, device="cpu"):
        ret_dic = {}
        for k, v in dic.items():
            if isinstance(v, list):
                v = torch.from_numpy(np.asarray(v)).to(device)
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                v = v.to(device)
            else:
                raise ValueError(f"unknown type: {type(v)}")
            ret_dic[k] = v
        return ret_dic
