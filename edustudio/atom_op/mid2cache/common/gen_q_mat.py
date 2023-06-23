from .base_mid2cache import BaseMid2Cache
import numpy as np
from itertools import chain
import torch


class M2C_GenQMat(BaseMid2Cache):
    def process(self, **kwargs):
        df_exer = kwargs['df_exer']
        cpt_count = len(set(list(chain(*df_exer['cpt_seq:token_seq'].to_list()))))
        # df_exer['cpt_multihot:token_seq'] = df_exer['cpt_seq:token_seq'].apply(
        #     lambda x: self.multi_hot(cpt_count, np.array(x)).tolist()
        # )
        kwargs['df_exer'] = df_exer
        tmp_df_exer = df_exer.set_index("exer_id:token")

        kwargs['Q_mat'] = torch.from_numpy(np.array(
            [self.multi_hot(cpt_count, tmp_df_exer.loc[exer_id]['cpt_seq:token_seq']).tolist()
              for exer_id in range(df_exer['exer_id:token'].max() + 1)]
        ))
        return kwargs

    @staticmethod
    def multi_hot(length, indices):
        multi_hot = np.zeros(length, dtype=np.int64)
        multi_hot[indices] = 1
        return multi_hot
