import numpy as np

from .kt_inter_datafmt_extends_q import KTInterDataFmtExtendsQ
from itertools import chain
import torch
import pandas as pd
from scipy import sparse

from ..utils.pad_seq_util import PadSeqUtil


class DIMKTDataFmt(KTInterDataFmtExtendsQ):
    default_cfg = {

    }

    def __init__(self,
                 cfg,
                 train_dict,
                 val_dict,
                 test_dict,
                 feat2type,
                 **kwargs
                ):
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type, **kwargs)
        # self.get_corr_data()
        self.train_dict = train_dict
        self.compute_difficulty()
        # self.construct_dif_seq()


    def compute_difficulty(self):
        q_dict = dict(zip(self.df_Q['exer_id'], self.df_Q['cpt_seq']))
        qd={}
        qd_count={}
        cd={}
        cd_count={}
        exer_ids = self.train_dict['exer_seq'].cpu().numpy()
        label_ids = self.train_dict['label_seq'].cpu().numpy()
        mask_ids = self.train_dict['mask_seq'].cpu().numpy()
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

        self.num_q = self.datafmt_cfg['dt_info']['exer_count']
        self.num_c = self.datafmt_cfg['dt_info']['cpt_count']
        self.q_dif = np.ones(self.num_q)
        self.c_dif = np.ones(self.num_c)
        for k,v in qd.items():
            self.q_dif[k] = int((qd[k]/qd_count[k])*100)+1
        for k,v in cd.items():
            self.c_dif[k] = int((cd[k]/cd_count[k])*100)+1
        self.q_dif = torch.tensor(self.q_dif).unsqueeze(1)
        self.c_dif = torch.tensor(self.c_dif).unsqueeze(1)

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['cpt_seq'] = torch.stack(
            [self.cpt_seq_padding[exer_seq][0] for exer_seq in dic['exer_seq']], dim=0
        )
        dic['cpt_seq_mask'] = torch.stack(
            [self.cpt_seq_mask[exer_seq][0] for exer_seq in dic['exer_seq']], dim=0
        )

        dic['qd_seq'] = torch.stack(
            [self.q_dif[exer_seq][0] for exer_seq in dic['exer_seq']], dim=0
        )
        dic['cd_seq'] = torch.stack(
            [self.c_dif[cpt_seq][0] for cpt_seq in dic['cpt_seq']], dim=0
        )
        mask = dic['mask_seq']==0
        dic['qd_seq'][mask]=0
        dic['cd_seq'][mask] = 0
        return dic



    @classmethod
    def construct_df2dict(cls, cfg, df: pd.DataFrame, maxlen=None):
        if df is None: return None

        tmp_df = df[['stu_id', 'exer_id', 'label', 'timestamp']].groupby('stu_id').agg(
            lambda x: list(x)
        ).reset_index()

        exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(
            tmp_df['exer_id'].to_list(), return_idx=True, return_mask=True,
            maxlen=maxlen
        )
        label_seq, _, _ = PadSeqUtil.pad_sequence(
            tmp_df['label'].to_list(), dtype=np.float32,
            maxlen=maxlen
        )

        time_seq, _, _, = PadSeqUtil.pad_sequence(
            tmp_df['timestamp'].to_list(), dtype=np.int64,
            maxlen=maxlen
        )

        stu_id = tmp_df['stu_id'].to_numpy()[idx]

        item_inputs = np.hstack((np.zeros((exer_seq.shape[0], 1)), exer_seq[:, :-1]))
        label_inputs = np.hstack((np.zeros((exer_seq.shape[0], 1)), label_seq[:, :-1]))

        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'label_seq': label_seq,
            'time_seq': time_seq,
            'item_inputs': item_inputs,
            'label_inputs': label_inputs,
            'mask_seq': mask_seq
        }
