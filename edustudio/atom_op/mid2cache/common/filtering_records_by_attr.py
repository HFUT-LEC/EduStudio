from .base_mid2cache import BaseMid2Cache
import pandas as pd
import numpy as np
from itertools import chain


class M2C_FilteringRecordsByAttr(BaseMid2Cache):
    """Commonly used by Fair Models, and Filtering Students without attribute values
    """
    default_cfg = {
        'filter_stu_attrs': ['gender:token']
    }

    def process(self, **kwargs):
        df_stu = kwargs['df_stu']
        df = kwargs['df']
        df_stu = df_stu[df_stu[self.m2c_cfg['filter_stu_attrs']].notna().all(axis=1)].reset_index(drop=True)
        df = df[df['stu_id:token'].isin(df_stu['stu_id:token'])].reset_index(drop=True)

        kwargs['df'] = df
        kwargs['df_stu'] = df_stu

        return kwargs
    





    