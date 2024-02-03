from ..common.base_mid2cache import BaseMid2Cache
import pandas as pd
import numpy as np
from edustudio.datatpl.utils import SpliterUtil, PadSeqUtil
from itertools import chain


class M2C_RandomDataSplit4KT(BaseMid2Cache):
    default_cfg = {
        'seed': 2023,
        'divide_by': 'stu',
        "divide_scale_list": [7,1,2],
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
        df_seq = kwargs['df_seq']
        df_train_seq = kwargs.get('df_train_seq', None)
        df_valid_seq = kwargs.get('df_validn_seq', None)
        df_test_seq = kwargs.get('df_test_seq', None)

        if not self.is_dataset_divided:
            assert df_train_seq is None and df_valid_seq is None and df_test_seq is None
            self.window_size = df_seq['exer_seq:token_seq'].shape[1]
            if self.m2c_cfg['divide_by'] == 'stu':
                if self.n_folds == 1:
                    train_dict, valid_dict, test_dict = self._divide_data_df_by_stu_one_fold(df_seq)
                    kwargs['df_train_folds'] = [train_dict]
                    kwargs['df_valid_folds'] = [valid_dict]
                    kwargs['df_test_folds'] = [test_dict]
                else:
                    kwargs['df_train_folds'], kwargs['df_valid_folds'], kwargs['df_test_folds'] = self._divide_data_df_by_stu_multi_fold(df_seq)
            elif self.m2c_cfg['divide_by'] == 'time':
                raise NotImplementedError
            else:
                raise ValueError(f"unknown divide_by: {self.m2c_cfg['divide_by']}")
        else:
            assert df_train_seq is not None and df_test_seq is not None
            self.window_size = df_train_seq['exer_seq:token_seq'].shape[1]
            kwargs['df_train_folds'] = [df_train_seq]
            kwargs['df_valid_folds'] = [df_valid_seq]
            kwargs['df_test_folds'] = [df_test_seq]
        return kwargs

    def _dict_index_flag(self, df_seq:dict, flag: np.array):
        return {
            k: df_seq[k][flag] for k in df_seq
        }

    def _divide_data_df_by_stu_one_fold(self, df_seq: dict):
        train_stu_id, valid_stu_id, test_stu_id = SpliterUtil.divide_data_df_one_fold(
            pd.DataFrame({"stu_id:token": np.unique(df_seq['stu_id:token'])}), seed=self.m2c_cfg['seed'], shuffle=True,
            divide_scale_list=self.m2c_cfg['divide_scale_list']
        )

        df_train_seq = self._dict_index_flag(df_seq, np.isin(df_seq['stu_id:token'], train_stu_id.to_numpy().flatten()))
        df_test_seq = self._dict_index_flag(df_seq, np.isin(df_seq['stu_id:token'], test_stu_id.to_numpy().flatten()))
        df_valid_seq = None
        if valid_stu_id is not None:
            df_valid_seq = self._dict_index_flag(df_seq, np.isin(df_seq['stu_id:token'], valid_stu_id.to_numpy().flatten()))

        return df_train_seq, df_test_seq, df_valid_seq
    
    def _divide_data_df_by_stu_multi_fold(self, df_seq: pd.DataFrame):
        res = SpliterUtil.divide_data_df_multi_folds(
            pd.DataFrame({"stu_id:token": np.unique(df_seq['stu_id:token'])}), seed=self.m2c_cfg['seed'], shuffle=True, n_folds=self.n_folds
        )

        train_list,  test_list = [], []
        for (train_stu_id, test_stu_id) in zip(*res):
            df_train_seq = self._dict_index_flag(df_seq, np.isin(df_seq['stu_id:token'], train_stu_id.to_numpy().flatten()))
            df_test_seq = self._dict_index_flag(df_seq, np.isin(df_seq['stu_id:token'], test_stu_id.to_numpy().flatten()))
            train_list.append(df_train_seq)
            test_list.append(df_test_seq)

        return train_list, [], test_list

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
