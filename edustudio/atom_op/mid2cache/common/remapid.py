from edustudio.utils.common import UnifyConfig
from .base_mid2cache import BaseMid2Cache
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import List
import numpy as np
from itertools import chain


class M2C_ReMapId(BaseMid2Cache):
    default_cfg = {
        'share_id_columns': [], # [{c1, c2, c3}]
        'ignore_columns': {"order_id:token"}
    }

    def __init__(self, m2c_cfg) -> None:
         super().__init__(m2c_cfg)
    
    def _check_params(self):
        for feat in self.m2c_cfg['ignore_columns']: assert 'token' in feat # 确保均是token特征

        # 检查share_id_columns合理性
        t = list(chain(*self.m2c_cfg['share_id_columns']))
        assert len(t) == len(set(t)), 'groups in share_id_columns should be disjoint'

    def process(self, **kwargs):
        super().process(**kwargs)

        feats = self.get_all_columns([v for v in kwargs.values() if type(v) is pd.DataFrame])
        feats = list(filter(lambda x: x.split(":")[-1] in {'token', 'token_seq'}, feats)) # 过滤掉非token特征
        feats = list(filter(lambda x: x not in self.m2c_cfg['ignore_columns'], feats)) # 过滤掉指定token特征
        feats_group = list(set(feats) - set(chain(*self.m2c_cfg['share_id_columns'])))
        feats_group = [set([v]) for v in feats_group]
        feats_group.extend(self.m2c_cfg['share_id_columns'])
        kwargs['feats_group'] = feats_group

        kwargs['lbe_dict'] = {}
        for feats in feats_group:
            col_arr = self.get_specific_column_into_arr(feats, kwargs.values())
            lbe = LabelEncoder().fit(col_arr)

            for v in kwargs.values():
                if type(v) is pd.DataFrame:
                    for col in feats:
                        if col in v.columns:
                            if col.split(":")[-1] == 'token':
                                v[col] = lbe.transform(v[col])
                            elif col.split(":")[-1] == 'token_seq':
                                v[col] = v[col].apply(lambda x: lbe.transform(x).tolist())
                            else:
                                raise ValueError("unsupport type of the feat: {col}")
            for f in feats:
                kwargs['lbe_dict'][f] = lbe
        return kwargs
                        
    @staticmethod
    def get_specific_column_into_arr(columns, df_list: List[pd.DataFrame]):
        col_list = []
        for v in df_list:
            if type(v) is pd.DataFrame:
                for col in columns:
                    if col in v.columns:
                        if col.split(":")[-1] == 'token':
                            col_list.append(v[col].to_numpy())
                        elif col.split(":")[-1] == 'token_seq':
                            col_list.extend(v[col].to_list())
                        else:
                            raise ValueError("unsupport type of the feat: {col}")
        col_arr = np.concatenate(col_list)
        return col_arr
    
    @staticmethod
    def get_all_columns(df_list: List[pd.DataFrame]):
        feats = set()
        for v in df_list: feats |= set(v.columns.tolist())
        return feats
