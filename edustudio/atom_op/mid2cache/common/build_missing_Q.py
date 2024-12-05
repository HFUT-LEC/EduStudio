from .base_mid2cache import BaseMid2Cache
import numpy as np
import pandas as pd
from itertools import chain
import torch
from edustudio.utils.common import set_same_seeds


class M2C_BuildMissingQ(BaseMid2Cache):
    default_cfg = {
        'seed': 20230518,
        'Q_delete_ratio': 0.0,
    }

    def process(self, **kwargs):
        dt_info = kwargs['dt_info']
        self.item_count = dt_info['exer_count']
        self.cpt_count = dt_info['cpt_count']
        self.df_Q = kwargs['df_exer'][['exer_id:token', 'cpt_seq:token_seq']]

        self.missing_df_Q = self.get_missing_df_Q()
        self.missing_Q_mat = self.get_Q_mat_from_df_arr(self.missing_df_Q, self.item_count, self.cpt_count)

        kwargs['missing_df_Q'] = self.missing_df_Q
        kwargs['missing_Q_mat'] = self.missing_Q_mat

        return kwargs

    def get_missing_df_Q(self):
        set_same_seeds(seed=self.m2c_cfg['seed'])
        ratio = self.m2c_cfg['Q_delete_ratio']
        iid2cptlist = self.df_Q.set_index('exer_id:token')['cpt_seq:token_seq'].to_dict()
        iid_lis = np.array(list(chain(*[[i]*len(iid2cptlist[i]) for i in iid2cptlist])))
        cpt_lis = np.array(list(chain(*list(iid2cptlist.values()))))
        entry_arr = np.vstack([iid_lis, cpt_lis]).T

        np.random.shuffle(entry_arr)

        # reference: https://stackoverflow.com/questions/64834655/python-how-to-find-first-duplicated-items-in-an-numpy-array
        _, idx = np.unique(entry_arr[:, 1], return_index=True) # 先从每个知识点中选出1题出来
        bool_idx = np.zeros_like(entry_arr[:, 1], dtype=bool)
        bool_idx[idx] = True
        preserved_exers = np.unique(entry_arr[bool_idx, 0]) # 选择符合条件的习题作为保留

        delete_num = int(ratio * self.item_count)
        preserved_num = self.item_count - delete_num

        if len(preserved_exers) >= preserved_num:
            self.logger.warning(
                f"Cant Satisfy Delete Require: {len(preserved_exers)=},{preserved_num=}"
            )
        else:
            need_preserved_num = preserved_num - len(preserved_exers)

            left_iids = np.arange(self.item_count)
            left_iids = left_iids[~np.isin(left_iids, preserved_exers)]
            np.random.shuffle(left_iids)
            choose_iids = left_iids[0:need_preserved_num]

            preserved_exers = np.hstack([preserved_exers, choose_iids])

        return self.df_Q.copy()[self.df_Q['exer_id:token'].isin(preserved_exers)].reset_index(drop=True)


    def get_Q_mat_from_df_arr(self, df_Q_arr, item_count, cpt_count):
        Q_mat = torch.zeros((item_count, cpt_count), dtype=torch.int64)
        for _, item in df_Q_arr.iterrows(): Q_mat[item['exer_id:token'], item['cpt_seq:token_seq']] = 1
        return Q_mat
