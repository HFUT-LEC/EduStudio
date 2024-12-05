from ..common import BaseMid2Cache
import torch
import numpy as np


class M2C_DIMKT_OP(BaseMid2Cache):
    default_cfg = {}

    def process(self, **kwargs):
        self.df_Q = kwargs['df_exer']
        dt_info = kwargs['dt_info']
        self.num_q = dt_info['exer_count']
        self.num_c = dt_info['cpt_count']

        df_train_folds = kwargs['df_train_folds']

        kwargs['q_diff_list'] = []
        kwargs['c_diff_list'] = []
        for train_dict in df_train_folds:
            self.train_dict = train_dict
            self.compute_difficulty()
            kwargs['q_diff_list'].append(self.q_dif)
            kwargs['c_diff_list'].append(self.c_dif)
        return kwargs

    def compute_difficulty(self):
        q_dict = dict(zip(self.df_Q['exer_id:token'], self.df_Q['cpt_seq:token_seq']))
        qd={}
        qd_count={}
        cd={}
        cd_count={}
        exer_ids = self.train_dict['exer_seq:token_seq']
        label_ids = self.train_dict['label_seq:float_seq']
        mask_ids = self.train_dict['mask_seq:token_seq']
        for ii, ee in enumerate(exer_ids):
            for i, e in enumerate(ee):
                tmp_mask = mask_ids[ii, i]
                if tmp_mask != 0:
                    tmp_exer = exer_ids[ii, i]
                    tmp_label = label_ids[ii, i]
                    cpt = (q_dict[tmp_exer])[0]
                    cd[cpt] = cd.get(cpt, 0) + tmp_label
                    cd_count[cpt] = cd_count.get(cpt, 0) + 1
                    if tmp_exer in qd:
                        qd[tmp_exer] = qd[tmp_exer] + tmp_label
                        qd_count[tmp_exer] = qd_count[tmp_exer]+1
                    else:
                        qd[tmp_exer] =  tmp_label
                        qd_count[tmp_exer] = 1
                else:
                    break


        self.q_dif = np.ones(self.num_q)
        self.c_dif = np.ones(self.num_c)
        for k,v in qd.items():
            self.q_dif[k] = int((qd[k]/qd_count[k])*100)+1
        for k,v in cd.items():
            self.c_dif[k] = int((cd[k]/cd_count[k])*100)+1
        self.q_dif = torch.tensor(self.q_dif).unsqueeze(1)
        self.c_dif = torch.tensor(self.c_dif).unsqueeze(1)
