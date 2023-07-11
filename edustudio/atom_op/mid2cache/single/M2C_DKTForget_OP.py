from ..common import BaseMid2Cache
import numpy as np
from collections import defaultdict


class M2C_DKTForget_OP(BaseMid2Cache):
    default_cfg = {
        'timestamp_unit': 'ms'
    }

    def process(self, **kwargs):
        df_train_folds = kwargs['df_train_folds']
        df_valid_folds = kwargs['df_valid_folds']
        df_test_folds = kwargs['df_test_folds']

        self.dt_info = kwargs['dt_info']
        self.dt_info['n_pcount_list'] = []
        self.dt_info['n_rgap_list'] = []
        self.dt_info['n_sgap_list'] = []

        for idx, (train_dict, test_dict) in enumerate(zip(df_train_folds, df_test_folds)):
            self.train_dict = train_dict
            self.test_dict = test_dict
            if df_valid_folds is not None and len(df_valid_folds) > 0:
                self.val_dict = df_valid_folds[idx]
                
            self.n_pcount, self.n_rgap, self.n_sgap = 0, 0, 0
            self._construct_s_gap(self.train_dict)
            self._construct_r_gap_and_p_count(self.train_dict)
            if self.val_dict:
                self._construct_s_gap(self.val_dict)
                self._construct_r_gap_and_p_count(self.val_dict)
            self._construct_s_gap(self.test_dict)
            self._construct_r_gap_and_p_count(self.test_dict)

            self.dt_info['n_pcount_list'].append(self.n_pcount)
            self.dt_info['n_rgap_list'].append(self.n_rgap)
            self.dt_info['n_sgap_list'].append(self.n_sgap)
        
        return kwargs

    @staticmethod
    def time2minutes(tensor, curr_unit):
        if curr_unit == 'ms':
            return tensor // (1000 * 60)
        elif curr_unit == 's':
            return tensor // 60
        elif curr_unit == 'minute':
            return tensor
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")

    def _construct_s_gap(self, data_dict):
        time_seq = data_dict['start_timestamp_seq:float_seq']
        mask_seq = data_dict['mask_seq:token_seq']
        s_gap = self.time2minutes(time_seq - np.hstack([time_seq[:,[0]], time_seq[:,0:-1]]), curr_unit=self.m2c_cfg['timestamp_unit'])
        s_gap[mask_seq == 0] = 1.0 # 防止log报错
        s_gap[:, 0] = 1.0 # 防止log报错
        assert (s_gap < 0).sum() == 0 # 确保时间是顺序的
        s_gap[s_gap == 0] = 1.0 # 防止log报错
        s_gap = np.log2(s_gap).astype(np.int64) + 1
        s_gap[mask_seq == 0] = 0 
        s_gap[:, 0] = 0 
        data_dict['s_gap'] = s_gap
        self.n_sgap = max(self.n_sgap, s_gap.max() + 1)

    def _construct_r_gap_and_p_count(self, data_dict):
        time_seq = data_dict['start_timestamp_seq:float_seq']
        mask_seq = data_dict['mask_seq:token_seq']
        exer_seq = data_dict['exer_seq:token_seq']

        p_count_mat = np.zeros(mask_seq.shape, dtype=np.float32)
        r_gap_mat = np.zeros(mask_seq.shape, dtype=np.float32)
        for seq_id, exer_seq_one in enumerate(exer_seq):
            exer_seq_mask_one = mask_seq[seq_id]
            time_seq_one = time_seq[seq_id]
            p_count_dict = defaultdict(int) # 记录习题被回答次数
            r_gap_dict = {} # 记录习题上次被回答时的时间
            for w_id, exer_id in enumerate(exer_seq_one[exer_seq_mask_one == 1]):
                exer_id = exer_id
                p_count_mat[seq_id, w_id] = p_count_dict[exer_id]
                p_count_dict[exer_id] += 1
                if exer_id in r_gap_dict:
                    r_gap_mat[seq_id, w_id] = self.time2minutes(time_seq_one[w_id] - r_gap_dict[exer_id],  curr_unit=self.m2c_cfg['timestamp_unit']) # 变为分钟
                    r_gap_dict[exer_id] = time_seq_one[w_id]
                else:
                    r_gap_dict[exer_id] = time_seq_one[w_id]
                    r_gap_mat[seq_id, w_id] = 0

        p_count_mat[:, 0] = 1.0
        p_count_mat[mask_seq == 0] = 1.0
        p_count_mat[p_count_mat == 0] = 1.0
        p_count_mat = np.log2(p_count_mat).astype(np.int64) + 1
        p_count_mat[:,0] = 0 # 这些单独作为一类
        p_count_mat[mask_seq == 0] = 0  # 这些单独作为一类
        
        r_gap_mat[:,0] = 1.0
        r_gap_mat[mask_seq == 0] = 1.0
        r_gap_mat[r_gap_mat == 0] = 1.0
        r_gap_mat = np.log2(r_gap_mat).astype(np.int64) + 1
        r_gap_mat[:,0] = 0
        r_gap_mat[mask_seq == 0] =  0

        data_dict['r_gap'] = r_gap_mat
        data_dict['p_count'] = p_count_mat

        self.n_pcount = max(self.n_pcount, p_count_mat.max()+ 1)
        self.n_rgap = max(self.n_rgap, r_gap_mat.max() + 1)
