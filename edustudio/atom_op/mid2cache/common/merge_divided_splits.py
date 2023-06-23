from .base_mid2cache import BaseMid2Cache
import pandas as pd
from itertools import chain


class M2C_MergeDividedSplits(BaseMid2Cache):
    default_cfg = {}

    def process(self, **kwargs):
        df_train = kwargs['df_train']
        df_valid = kwargs['df_valid']
        df_test = kwargs['df_test']
        
        # 1. ensure the keys in df_train, df_valid, df_test is same
        assert df_train is not None and df_test is not None
        assert set(df_train.columns) == set(df_test.columns)
        # 2. 包容df_valid不存在的情况

        df = pd.concat((df_train, df_test), ignore_index=True)

        if df_valid is not None:
            assert set(df_train.columns) == set(df_valid.columns)
            df = pd.concat((df, df_valid), ignore_index=True)
        
        kwargs['df'] = df
        return kwargs

    def set_dt_info(self, dt_info, **kwargs):
        if 'stu_id:token' in kwargs['df'].columns:
            dt_info['stu_count'] = int(kwargs['df']['stu_id:token'].max() + 1)
        if 'exer_id:token' in kwargs['df'].columns:
            dt_info['exer_count'] = int(kwargs['df']['exer_id:token'].max() + 1)
        if kwargs.get('df_exer', None) is not None:
            if 'cpt_seq:token_seq' in kwargs['df_exer']:
                dt_info['cpt_count'] = len(set(list(chain(*kwargs['df_exer']['cpt_seq:token_seq'].to_list()))))
