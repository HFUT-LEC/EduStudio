from .base_mid2cache import BaseMid2Cache
import numpy as np


class M2C_Label2Int(BaseMid2Cache):
    def process(self, **kwargs):
        self.op_on_df('df', kwargs)
        self.op_on_df('df_train', kwargs)
        self.op_on_df('df_valid', kwargs)
        self.op_on_df('df_test', kwargs)
        return kwargs

    @staticmethod
    def op_on_df(column, kwargs):
        if column in kwargs and kwargs[column] is not None:
            kwargs[column]['label:float'] =  (kwargs[column]['label:float'] >= 0.5).astype(np.float32)
