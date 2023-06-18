from .kt_inter_datafmt_extends_q import KTInterDataFmtExtendsQ
import numpy as np
from ..utils import PadSeqUtil
import pandas as pd
import torch
from collections import defaultdict


class HawkesKTDataFmt(KTInterDataFmtExtendsQ):
    default_cfg = {
        'timestamp_unit': 'ms'
    }
    def __init__(self, cfg, train_dict, val_dict, test_dict, feat2type, **kwargs):
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type, **kwargs)
        
    def _init_data_after_dt_info(self):
        super()._init_data_after_dt_info()
        self.construct_forget_feats()

    def construct_forget_feats(self):
        self._construct_s_gap(self.train_dict)
        self._construct_r_gap_and_p_count(self.train_dict)
        if self.val_dict:
            self._construct_s_gap(self.val_dict)
            self._construct_r_gap_and_p_count(self.val_dict)
        self._construct_s_gap(self.test_dict)
        self._construct_r_gap_and_p_count(self.test_dict)

    # def _stat_dataset_info(self):
    #     super()._stat_dataset_info()
    #     exer_count = max(self.datafmt_cfg['dt_info']['exer_count'], self.df_Q['exer_id'].max()+1)
    #     self.datafmt_cfg['dt_info']['exer_count'] = exer_count
        
    #     cpt_count = len(set(list(chain(*self.df_Q['cpt_seq'].to_list()))))
    #     self.datafmt_cfg['dt_info']['cpt_count'] = cpt_count

    @staticmethod
    def time2seconds(tensor, curr_unit, rounding_mode='floor'):
        if curr_unit == 'ms':
            return torch.div(tensor, 1000 , rounding_mode=rounding_mode)
        elif curr_unit == 's':
            return tensor
        elif curr_unit == 'minute':
            return torch.div(tensor, 1/60, rounding_mode=rounding_mode)
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")

    def _construct_s_gap(self, data_dict):
        time_seq = data_dict['time_seq']
        mask_seq = data_dict['mask_seq']
        s_gap = self.time2seconds(time_seq - time_seq[:,[0]], curr_unit=self.datafmt_cfg['timestamp_unit'])
        # torch.hstack是水平拼接两个tensor
        s_gap[mask_seq == 0] = 1.0 # 防止log报错
        s_gap[:, 0] = 1.0 # 防止log报错
        assert (s_gap < 0).sum() == 0 # 确保时间是顺序的
        s_gap[s_gap == 0] = 1.0 # 防止log报错
        s_gap = torch.log2(s_gap).long() + 1
        s_gap[mask_seq == 0] = 0 
        s_gap[:, 0] = 0 
        data_dict['s_gap'] = s_gap
        self.datafmt_cfg['dt_info']['n_sgap'] =  max(self.datafmt_cfg['dt_info'].get('n_sgap', 0), s_gap.max().item() + 1)

    @classmethod
    def construct_df2dict(cls, cfg, df: pd.DataFrame, maxlen=None):
        if df is None: return None

        tmp_df = df[['stu_id','exer_id','label', 'timestamp']].groupby('stu_id').agg(
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
        
        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'label_seq': label_seq,
            'mask_seq': mask_seq,
            'time_seq': time_seq
        }
