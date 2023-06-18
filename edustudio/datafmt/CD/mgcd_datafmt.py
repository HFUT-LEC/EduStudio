from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
import pandas as pd
from typing import Dict
import torch
from itertools import chain
import os
from ..utils import SpliterUtil


class MgcdDataFmt(CDInterDataFmtExtendsQ):
    def __init__(self, cfg, train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], feat_name2type: Dict[str, str], 
                 df_Q, val_dict: Dict[str, torch.tensor] = None
                 ):
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)
        self.read_inter_student(cfg)
        self.read_G(cfg)
    
    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)
        name2type, df_Q = cls.read_Q_matrix(cfg)
        feat_name2type.update(name2type)
        train_dict=cls.df2dict(train_df)
        train_dict['stu_id'] = train_dict['group_id']
        del train_dict['group_id']
        test_dict=cls.df2dict(test_df)
        test_dict['stu_id'] = test_dict['group_id']
        del test_dict['group_id']
        val_dict = None
        if val_df is not None:
            val_dict=cls.df2dict(val_df)
            val_dict['stu_id'] = val_dict['group_id']
            del val_dict['group_id']
        return cls(
            cfg=cfg,
            train_dict=train_dict,
            test_dict=test_dict,
            val_dict=val_dict,
            df_Q=df_Q,
            feat_name2type=feat_name2type
        )

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
            'group_count': self.stu_count,
            'exer_count': self.exer_count,
            'trainset_count': self.train_num,
            'valset_count': self.val_num,
            'testset_count': self.test_num
        })
        exer_count = max(self.datafmt_cfg['dt_info']['exer_count'], self.df_Q['exer_id'].max()+1)
        self.datafmt_cfg['dt_info']['exer_count'] = exer_count
        
        cpt_count = len(set(list(chain(*self.df_Q['cpt_seq'].to_list()))))
        self.datafmt_cfg['dt_info']['cpt_count'] = cpt_count

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
                    seed=cfg.datafmt_cfg['seed'], label_field='group_id'
                )
            else:
                raise NotImplementedError

        return feat_name2type, train_df, val_df, test_df
    
    @classmethod
    def _read_data_from_undivided(cls, cfg):  # CDInterDataFmt里运行这个函数会调用这个方法
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}.inter_group.csv'
        headers = pd.read_csv(file_path, nrows=0).columns.tolist()
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']
        inter_df = pd.read_csv(
            file_path, sep=cfg.datafmt_cfg['seperator'], encoding='utf-8',
            usecols=set(headers) - set(exclude_feats)
        )
        return inter_df

    def read_inter_student(self, cfg):  # assist-2012.inter_student.csv
        inter_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}.inter_student.csv'
        sep = cfg.datafmt_cfg['seperator']
        headers = pd.read_csv(inter_file_path, nrows=0).columns.tolist()
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']
        inter_df = pd.read_csv(
            inter_file_path, sep=sep, encoding='utf-8', usecols=set(headers) - set(exclude_feats)
        )
        feat_name2type = {}
        for col in inter_df.columns:
            col_name, col_type = col.split(":")
            feat_name2type[col_name] = col_type
            if col_type == 'token':
                inter_df[col] = inter_df[col].astype('int64')
            elif col_type == 'float':
                inter_df[col] = inter_df[col].astype('float32')
            elif col_type == 'token_seq':
                pass
            elif col_type == 'float_seq':
                pass
            else:
                raise ValueError(f"unknown field type of {col_type}")
            inter_df.rename(columns={col: col_name}, inplace=True)
        self.inter_student = {k: torch.from_numpy(inter_df[k].to_numpy()) for k in inter_df}

    def read_G(self, cfg): # assist-2012.G.csv
        G_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}.G.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_G = pd.read_csv(G_file_path, sep=sep, encoding='utf-8', usecols=['group_id:token', 'stu_seq:token_seq'])
        feat_name2type = {}
        for col in df_G.columns:
            col_name, col_type = col.split(":")
            feat_name2type[col_name] = col_type
            if col_type == 'token':
                df_G[col] = df_G[col].astype('int64')
            elif col_type == 'float':
                df_G[col] = df_G[col].astype('float32')
            elif col_type == 'token_seq':
                pass
            elif col_type == 'float_seq':
                pass
            else:
                raise ValueError(f"unknown field type of {col_type}")
            df_G.rename(columns={col: col_name}, inplace=True)

        df_G['stu_seq'] = df_G['stu_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        self.df_G = df_G
    
    def get_extra_data(self):
        # assist-2012.inter_student.csv assist-2012.G.csv
        return {
            'inter_student': self.inter_student,
            'df_G': self.df_G
        }
