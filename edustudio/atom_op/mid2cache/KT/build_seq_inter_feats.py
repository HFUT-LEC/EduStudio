from ..common.base_mid2cache import BaseMid2Cache
import pandas as pd
import numpy as np
from edustudio.datatpl.utils import SpliterUtil, PadSeqUtil
from itertools import chain


class M2C_BuildSeqInterFeats(BaseMid2Cache):
    default_cfg = {
        'seed': 2023,
        'divide_by': 'stu',
        'window_size': 100,
        "divide_scale_list": [7,1,2],
        "extra_inter_feats": []
    }

    def __init__(self, m2c_cfg, n_folds, is_dataset_divided) -> None:
        super().__init__(m2c_cfg)
        self.n_folds = n_folds
        self.is_dataset_divided = is_dataset_divided

    @classmethod
    def from_cfg(cls, cfg):
        m2c_cfg = cfg.datatpl_cfg.get(cls.__name__)
        n_folds = cfg.datatpl_cfg.n_folds
        is_dataset_divided = cfg.datatpl_cfg.is_dataset_divided
        return cls(m2c_cfg, n_folds, is_dataset_divided)

    def _check_params(self):
        super()._check_params()
        assert self.m2c_cfg['divide_by'] in {'stu', 'time'}

    def process(self, **kwargs):
        df = kwargs['df']
        df_train, df_valid, df_test = kwargs['df_train'], kwargs['df_valid'], kwargs['df_test']
        df = self.sort_records(df)
        df_train = self.sort_records(df_train)
        df_valid = self.sort_records(df_valid)
        df_test = self.sort_records(df_test)

        if not self.is_dataset_divided:
            assert df_train is None and df_valid is None and df_test is None
            if self.m2c_cfg['divide_by'] == 'stu':
                if self.n_folds == 1:
                    train_dict, valid_dict, test_dict = self._divide_data_df_by_stu_one_fold(df)
                    kwargs['df_train_folds'] = [train_dict]
                    kwargs['df_valid_folds'] = [valid_dict]
                    kwargs['df_test_folds'] = [test_dict]
                else:
                    kwargs['df_train_folds'], kwargs['df_valid_folds'], kwargs['df_test_folds'] = self._divide_data_df_by_stu_multi_fold(df)
            elif self.m2c_cfg['divide_by'] == 'time':
                raise NotImplementedError
            else:
                raise ValueError(f"unknown divide_by: {self.m2c_cfg['divide_by']}")
        else: # dataset is divided
            assert df_train is not None and df_test is not None
            if self.m2c_cfg['window_size'] <= 0 or self.m2c_cfg['window_size'] is None:
                self.window_size = np.max([
                    df_train[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max(),
                    df_valid[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max() if df_valid is not None else 0,
                    df_valid[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max()
                ])
                self.logger.info(f"actual window size: {self.window_size}")
            else:
                self.window_size = self.m2c_cfg['window_size']
            train_dict = self.construct_df2dict(df_train)
            valid_dict = self.construct_df2dict(df_valid)
            test_dict = self.construct_df2dict(df_test)
            kwargs['df_train_folds'] = [train_dict]
            kwargs['df_valid_folds'] = [valid_dict]
            kwargs['df_test_folds'] = [test_dict]
        return kwargs

    @staticmethod
    def sort_records(df, col='order_id:token'):
        if df is not None:
            return df.sort_values(by=col, ascending=True).reset_index(drop=True)

    def _divide_data_df_by_stu_one_fold(self, df: pd.DataFrame):
        train_stu_id, val_stu_id, test_stu_id = SpliterUtil.divide_data_df_one_fold(
            df['stu_id:token'].drop_duplicates(), seed=self.m2c_cfg['seed'], shuffle=True,
            divide_scale_list=self.m2c_cfg['divide_scale_list']
        )
        train_df = df[df['stu_id:token'].isin(train_stu_id)]
        val_df = df[df['stu_id:token'].isin(val_stu_id)] if val_stu_id is not None else None
        test_df = df[df['stu_id:token'].isin(test_stu_id)]
        
        if self.m2c_cfg['window_size'] <= 0 or self.m2c_cfg['window_size'] is None:
            self.window_size = np.max([
                train_df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max(),
                val_df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max() if val_df is not None else 0,
                test_df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max()
            ])
            self.logger.info(f"actual window size: {self.window_size}")
        else:
            self.window_size = self.m2c_cfg['window_size']
        
        train_dict = self.construct_df2dict(train_df)
        val_dict = self.construct_df2dict(val_df)
        test_dict = self.construct_df2dict(test_df)
        return train_dict, val_dict, test_dict
    
    def _divide_data_df_by_stu_multi_fold(self, df: pd.DataFrame):
        res = SpliterUtil.divide_data_df_one_fold(
            df['stu_id:token'].drop_duplicates(), seed=self.m2c_cfg['seed'], shuffle=True,
            divide_scale_list=self.m2c_cfg['divide_scale_list']
        )

        train_list, valid_list, test_list = [], [], []
        for train_stu_id, val_stu_id, test_stu_id in zip(res):
            train_df = df[df['stu_id:token'].isin(train_stu_id)]
            val_df = df[df['stu_id:token'].isin(val_stu_id)] if val_stu_id is not None else None
            test_df = df[df['stu_id:token'].isin(test_stu_id)]
            
            if self.m2c_cfg['window_size'] <= 0 or self.m2c_cfg['window_size'] is None:
                self.window_size = np.max([
                    train_df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max(),
                    val_df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max() if val_df is not None else 0,
                    test_df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max()
                ])
                self.logger.info(f"actual window size: {self.window_size}")
            else:
                self.window_size = self.m2c_cfg['window_size']
            
            train_dict = self.construct_df2dict(train_df)
            valid_dict = self.construct_df2dict(val_df)
            test_dict = self.construct_df2dict(test_df)
            train_list.append(train_dict)
            valid_list.append(valid_dict)
            test_list.append(test_dict)

        return train_list, valid_list, test_list
    
    def construct_df2dict(self, df: pd.DataFrame):
        if df is None: return None

        tmp_df = df[['stu_id:token','exer_id:token','label:float'] + self.m2c_cfg['extra_inter_feats']].groupby('stu_id:token').agg(lambda x: list(x)).reset_index()

        exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(
            tmp_df['exer_id:token'].to_list(), return_idx=True, return_mask=True, 
            maxlen=self.window_size
        )
        label_seq, _, _ = PadSeqUtil.pad_sequence(
            tmp_df['label:float'].to_list(), dtype=np.float32,
            maxlen=self.window_size
        )
        stu_id = tmp_df['stu_id:token'].to_numpy()[idx]

        ret_dict = {
            'stu_id:token': stu_id,
            'exer_seq:token_seq': exer_seq,
            'label_seq:float_seq': label_seq,
            'mask_seq:token_seq': mask_seq
        }
        for extra_feat in self.m2c_cfg['extra_inter_feats']:
            name, type_ = extra_feat.split(":")
            if type_ == 'token':
                seq, _, _ = PadSeqUtil.pad_sequence(
                    tmp_df[extra_feat].to_list(), dtype=np.int64,
                    maxlen=self.window_size
                )
                ret_dict[f"{name}_seq:token_seq"] = seq
            elif type_ == 'float':
                seq, _, _ = PadSeqUtil.pad_sequence(
                    tmp_df[extra_feat].to_list(), dtype=np.float32,
                    maxlen=self.window_size
                )
                ret_dict[f"{name}_seq:float_seq"] = seq
            else:
                raise NotImplementedError

        return ret_dict

    def set_dt_info(self, dt_info, **kwargs):
        dt_info['real_window_size'] = self.window_size
        if not self.is_dataset_divided:
            if 'stu_id:token' in kwargs['df'].columns:
                dt_info['stu_count'] = int(kwargs['df']['stu_id:token'].max() + 1)
            if 'exer_id:token' in kwargs['df'].columns:
                dt_info['exer_count'] = int(kwargs['df']['exer_id:token'].max() + 1)
        else:
            stu_count = max(kwargs['df_train']['stu_id:token'].max() + 1, kwargs['df_test']['stu_id:token'].max() + 1)
            stu_count = max(kwargs['df_valid']['stu_id:token'].max() + 1, stu_count) if 'df_valid' in kwargs else stu_count

            exer_count = max(kwargs['df_train']['exer_id:token'].max() + 1, kwargs['df_test']['exer_id:token'].max() + 1)
            exer_count = max(kwargs['df_valid']['exer_id:token'].max() + 1, exer_count) if 'df_valid' in kwargs else exer_count

            dt_info['stu_count'] = stu_count
            dt_info['exer_count'] = exer_count

        if kwargs.get('df_exer', None) is not None:
            if 'cpt_seq:token_seq' in kwargs['df_exer']:
                dt_info['cpt_count'] = len(set(list(chain(*kwargs['df_exer']['cpt_seq:token_seq'].to_list()))))
