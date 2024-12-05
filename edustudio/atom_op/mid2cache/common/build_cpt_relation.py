from .base_mid2cache import BaseMid2Cache
import pandas as pd
import numpy as np
from itertools import chain


class M2C_BuildKCRelation(BaseMid2Cache):
    default_cfg = {
        'relation_type': 'rcd_transition',
        'threshold': None
    }

    def process(self, **kwargs):
        df = kwargs['df']
        if df is None:
            df = pd.concat(
                [kwargs['df_train'], kwargs['df_valid'], kwargs['df_test']],
                axis=0, ignore_index=True
            )

        if 'order_id:token' in df:
            df.sort_values(by=['order_id:token'], axis=0, ignore_index=True, inplace=True)
        
        kwargs['df'] = df

        if self.m2c_cfg['relation_type'] == 'rcd_transition':
            return self.gen_rcd_transition_relation(kwargs)
        else:
            raise NotImplementedError
        
    def gen_rcd_transition_relation(self, kwargs):
        df:pd.DataFrame = kwargs['df']
        df_exer = kwargs['df_exer'][['exer_id:token', 'cpt_seq:token_seq']]
        df = df[['stu_id:token', 'exer_id:token', 'label:float']].merge(df_exer[['exer_id:token', 'cpt_seq:token_seq']], how='left', on='exer_id:token')
        cpt_count = np.max(list(chain(*df_exer['cpt_seq:token_seq'].to_list()))) + 1
        
        n_mat = np.zeros((cpt_count, cpt_count), dtype=np.float32) # n_{i,j}
        for _, df_one_stu in df.groupby('stu_id:token'):
            for idx in range(df_one_stu.shape[0]):
                if idx == df_one_stu.shape[0] - 2:
                    break
                curr_record = df_one_stu.iloc[idx]
                next_record = df_one_stu.iloc[idx + 1]
                if curr_record['label:float'] * next_record['label:float'] == 1:
                    for cpt_pre in curr_record['cpt_seq:token_seq']:
                        for cpt_next in next_record['cpt_seq:token_seq']:
                            if cpt_pre != cpt_next:
                                n_mat[cpt_pre, cpt_next] += 1

        a = np.sum(n_mat, axis=1)[:,None]
        nonzero_mask = (a != 0)
        np.seterr(divide='ignore', invalid='ignore')
        C_mat = np.where(nonzero_mask, n_mat / a, n_mat)

        max_val = C_mat.max()
        np.fill_diagonal(C_mat, max_val)
        min_val = C_mat.min()
        np.fill_diagonal(C_mat, 0)
        T_mat = (C_mat- min_val) / (max_val - min_val)

        threshold = self.m2c_cfg['threshold']
        if threshold is None:
            threshold = C_mat.sum() / (C_mat != 0).sum()
            threshold *= threshold
            threshold *= threshold
        
        cpt_dep_mat = (T_mat > threshold).astype(np.int64)

        kwargs['cpt_dep_mat'] = cpt_dep_mat

        return kwargs
