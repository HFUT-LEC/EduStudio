from ..common.base_mid2cache import BaseMid2Cache
import pandas as pd
import numpy as np
from edustudio.datatpl.utils import SpliterUtil, PadSeqUtil
from itertools import chain


class M2C_BuildSeqInterFeats(BaseMid2Cache):
    default_cfg = {
        'window_size': 100,
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
    
    def process(self, **kwargs):
        df = kwargs['df']
        df_train, df_valid, df_test = kwargs['df_train'], kwargs['df_valid'], kwargs['df_test']
        df = self.sort_records(df)
        df_train = self.sort_records(df_train)
        df_valid = self.sort_records(df_valid)
        df_test = self.sort_records(df_test)

        if not self.is_dataset_divided:
            assert df_train is None and df_valid is None and df_test is None
            self.window_size = self.m2c_cfg['window_size']
            if self.m2c_cfg['window_size'] <= 0 or self.m2c_cfg['window_size'] is None:
                self.window_size = df[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max()
            self.logger.info(f"actual window size: {self.window_size}")
            kwargs['df_seq'] = self.construct_df2dict(df)

        else: # dataset is divided
            assert df_train is not None and df_test is not None
            if self.m2c_cfg['window_size'] <= 0 or self.m2c_cfg['window_size'] is None:
                self.window_size = np.max([
                    df_train[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max(),
                    df_valid[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max() if df_valid is not None else 0,
                    df_test[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg('count')['exer_id:token'].max()
                ])
            else:
                self.window_size = self.m2c_cfg['window_size']
            self.logger.info(f"actual window size: {self.window_size}")

            train_dict = self.construct_df2dict(df_train)
            valid_dict = self.construct_df2dict(df_valid)
            test_dict = self.construct_df2dict(df_test)
            kwargs['df_train_seq'] = train_dict
            kwargs['df_valid_seq'] = valid_dict
            kwargs['df_test_seq'] = test_dict
        return kwargs

    @staticmethod
    def sort_records(df, col='order_id:token'):
        if df is not None:
            return df.sort_values(by=col, ascending=True).reset_index(drop=True)
    
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
