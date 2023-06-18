from .kt_inter_datafmt_extends_q import KTInterDataFmtExtendsQ
import numpy as np
from ..utils import PadSeqUtil
import pandas as pd
import torch
from collections import defaultdict


class DKTForgetDataFmt(KTInterDataFmtExtendsQ):
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

    @staticmethod
    def time2minutes(tensor, curr_unit, rounding_mode='floor'):
        if curr_unit == 'ms':
            return torch.div(tensor, 1000 * 60, rounding_mode=rounding_mode)
        elif curr_unit == 's':
            return torch.div(tensor, 60, rounding_mode=rounding_mode)
        elif curr_unit == 'minute':
            return tensor
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")

    def _construct_s_gap(self, data_dict):
        time_seq = data_dict['time_seq']
        mask_seq = data_dict['mask_seq']
        s_gap = self.time2minutes(time_seq - torch.hstack([time_seq[:,[0]], time_seq[:,0:-1]]), curr_unit=self.datafmt_cfg['timestamp_unit'])
        s_gap[mask_seq == 0] = 1.0 # 防止log报错
        s_gap[:, 0] = 1.0 # 防止log报错
        assert (s_gap < 0).sum() == 0 # 确保时间是顺序的
        s_gap[s_gap == 0] = 1.0 # 防止log报错
        s_gap = torch.log2(s_gap).long() + 1
        s_gap[mask_seq == 0] = 0 
        s_gap[:, 0] = 0 
        data_dict['s_gap'] = s_gap
        self.datafmt_cfg['dt_info']['n_sgap'] =  max(self.datafmt_cfg['dt_info'].get('n_sgap', 0), s_gap.max().item() + 1)

    def _construct_r_gap_and_p_count(self, data_dict):
        time_seq = data_dict['time_seq']
        mask_seq = data_dict['mask_seq']
        exer_seq = data_dict['exer_seq']

        p_count_mat = torch.zeros(mask_seq.shape, dtype=torch.float32)
        r_gap_mat = torch.zeros(mask_seq.shape, dtype=torch.float32)
        for seq_id, exer_seq_one in enumerate(exer_seq):
            exer_seq_mask_one = mask_seq[seq_id]
            time_seq_one = time_seq[seq_id]
            p_count_dict = defaultdict(int) # 记录习题被回答次数
            r_gap_dict = {} # 记录习题上次被回答时的时间
            for w_id, exer_id in enumerate(exer_seq_one[exer_seq_mask_one == 1]):
                exer_id = exer_id.item()
                p_count_mat[seq_id, w_id] = p_count_dict[exer_id]
                p_count_dict[exer_id] += 1
                if exer_id in r_gap_dict:
                    r_gap_mat[seq_id, w_id] = self.time2minutes(time_seq_one[w_id] - r_gap_dict[exer_id],  curr_unit=self.datafmt_cfg['timestamp_unit']) # 变为分钟
                    r_gap_dict[exer_id] = time_seq_one[w_id]
                else:
                    r_gap_dict[exer_id] = time_seq_one[w_id]
                    r_gap_mat[seq_id, w_id] = 0

        p_count_mat[:, 0] = 1.0
        p_count_mat[mask_seq == 0] = 1.0
        p_count_mat[p_count_mat == 0] = 1.0
        p_count_mat = torch.log2(p_count_mat).long() + 1
        p_count_mat[:,0] = 0 # 这些单独作为一类
        p_count_mat[mask_seq == 0] = 0  # 这些单独作为一类
        
        r_gap_mat[:,0] = 1.0
        r_gap_mat[mask_seq == 0] = 1.0
        r_gap_mat[r_gap_mat == 0] = 1.0
        r_gap_mat = torch.log2(r_gap_mat).long() + 1
        r_gap_mat[:,0] = 0
        r_gap_mat[mask_seq == 0] =  0

        data_dict['r_gap'] = r_gap_mat
        data_dict['p_count'] = p_count_mat
        self.datafmt_cfg['dt_info']['n_pcount'] = max(self.datafmt_cfg['dt_info'].get('n_pcount', 0), p_count_mat.max().item() + 1)
        self.datafmt_cfg['dt_info']['n_rgap'] = max(self.datafmt_cfg['dt_info'].get('n_rgap', 0), r_gap_mat.max().item() + 1)

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
