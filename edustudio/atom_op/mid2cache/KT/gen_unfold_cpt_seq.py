from ..common.base_mid2cache import BaseMid2Cache
import numpy as np
from edustudio.datatpl.utils import PadSeqUtil
import pandas as pd


class M2C_GenUnFoldKCSeq(BaseMid2Cache):
    default_cfg = {}
    
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
        df_exer = kwargs['df_exer']
        df_train, df_valid, df_test = kwargs['df_train'], kwargs['df_valid'], kwargs['df_test']


        if not self.is_dataset_divided:
            assert df_train is None and df_valid is None and df_test is None
            unique_cpt_seq = df_exer['cpt_seq:token_seq'].explode().unique()
            cpt_map = dict(zip(unique_cpt_seq, range(len(unique_cpt_seq))))
            df_Q_unfold = pd.DataFrame({
                'exer_id:token': df_exer['exer_id:token'].repeat(df_exer['cpt_seq:token_seq'].apply(len)),
                'cpt_unfold:token': df_exer['cpt_seq:token_seq'].explode().replace(cpt_map)
            })
            df = pd.merge(df, df_Q_unfold, on=['exer_id:token'], how='left').reset_index(drop=True)
            kwargs['df'] = df
        else: # dataset is divided
            assert df_train is not None and df_test is not None
            train_df, val_df, test_df = self._unfold_dataset(df_train, df_valid, df_test, df_exer)
            kwargs['df_train'] = train_df
            kwargs['df_valid'] = val_df
            kwargs['df_test'] = test_df
        return kwargs

    def _unfold_dataset(self, train_df, val_df, test_df, df_Q):
        unique_cpt_seq = df_Q['cpt_seq'].explode().unique()
        cpt_map = dict(zip(unique_cpt_seq, range(len(unique_cpt_seq))))
        df_Q_unfold = pd.DataFrame({
            'exer_id:token': df_Q['exer_id'].repeat(df_Q['cpt_seq'].apply(len)),
            'cpt_unfold:token': df_Q['cpt_seq'].explode().replace(cpt_map)
        })
        train_df_unfold = pd.merge(train_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)
        val_df_unfold = pd.merge(val_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)
        test_df_unfold = pd.merge(test_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)

        return train_df_unfold, val_df_unfold, test_df_unfold
