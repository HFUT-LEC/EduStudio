from .kt_inter_datafmt_extends_q import KTInterDataFmtExtendsQ
from ..utils import PadSeqUtil
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


class LPKTDataFmt(KTInterDataFmtExtendsQ):
    default_cfg = {
        'answer_time_unit': 's', # ['ms', 's', 'minute']
        'timestamp_unit': 's', # ['ms', 's', 'minute']
    }
    def _init_data_after_dt_info(self):
        super()._init_data_after_dt_info()
        self.construct_answer_time()
        self.construct_interval_time()

    def get_extra_data(self):
        return {
            "Q_mat": self.Q_mat
        }
    
    def __getitem__(self, index):
        dic = super().__getitem__(index)
        return dic
    
    def construct_answer_time(self):
        # 1. train, val, test中所有answer_time变成秒
        curr_unit = self.datafmt_cfg['answer_time_unit']
        answer_time = []
        answer_time.append(self.answer_time2seconds(self.train_dict['answer_time_seq'], curr_unit))
        if self.val_dict is not None:
            answer_time.append(self.answer_time2seconds(self.val_dict['answer_time_seq'], curr_unit))
        answer_time.append(self.answer_time2seconds(self.test_dict['answer_time_seq'], curr_unit))

        # 2. 离散化
        at_lbe = LabelEncoder()
        at_lbe.fit(torch.concat(answer_time).flatten().numpy())
        self.train_dict['answer_time_seq'] = torch.from_numpy(at_lbe.transform(
            self.train_dict['answer_time_seq'].flatten().numpy()).reshape(self.train_dict['answer_time_seq'].shape))
        if self.val_dict is not None:
            self.val_dict['answer_time_seq'] = torch.from_numpy(at_lbe.transform(
                self.val_dict['answer_time_seq'].flatten().numpy()).reshape(self.val_dict['answer_time_seq'].shape))
        self.test_dict['answer_time_seq'] = torch.from_numpy(at_lbe.transform(
            self.test_dict['answer_time_seq'].flatten().numpy()).reshape(self.test_dict['answer_time_seq'].shape))

        self.datafmt_cfg['dt_info']['answer_time_count'] = len(at_lbe.classes_)

    def construct_interval_time(self):
        # 1. 计算train, val, test中的interval_time, 将interval time变成分钟
        curr_unit = self.datafmt_cfg['timestamp_unit']
        interval_time = []
        train_it = self.train_dict['time_seq'] - torch.cat([self.train_dict['time_seq'][:,[0]], self.train_dict['time_seq'][:,0:-1]], dim=1)
        train_it[train_it < 0] = 0
        train_it = self.interval_time2minutes(train_it, curr_unit)
        train_it[train_it > 43200] = 43200
        interval_time.append(train_it)
        test_it = self.test_dict['time_seq'] - torch.cat([self.test_dict['time_seq'][:,[0]], self.test_dict['time_seq'][:,0:-1]], dim=1)
        test_it[test_it < 0] = 0
        test_it = self.interval_time2minutes(test_it, curr_unit)
        test_it[test_it > 43200] = 43200
        interval_time.append(test_it)
        if self.val_dict is not None:
            val_it = self.val_dict['time_seq'] - torch.cat([self.val_dict['time_seq'][:,[0]], self.val_dict['time_seq'][:,0:-1]], dim=1)
            val_it[val_it < 0] = 0
            val_it = self.interval_time2minutes(val_it, curr_unit)
            val_it[val_it > 43200] = 43200
            interval_time.append(val_it)
        
        # 2. 离散化
        it_lbe = LabelEncoder()
        it_lbe.fit(torch.concat(interval_time).flatten().numpy())
        self.train_dict['interval_time_seq'] = torch.from_numpy(
            it_lbe.transform(train_it.flatten().numpy()).reshape(train_it.shape))
        if self.val_dict is not None:
            self.val_dict['interval_time_seq'] = torch.from_numpy(
                it_lbe.transform(val_it.flatten().numpy()).reshape(val_it.shape))
        self.test_dict['interval_time_seq'] = torch.from_numpy(
            it_lbe.transform(test_it.flatten().numpy()).reshape(test_it.shape))

        self.datafmt_cfg['dt_info']['interval_time_count'] = len(it_lbe.classes_)

    @staticmethod
    def answer_time2seconds(tensor, curr_unit):
        if curr_unit == 'ms':
            return torch.div(tensor, 1000, rounding_mode='trunc')
        elif curr_unit == 's':
            return tensor
        elif curr_unit == 'minute':
            return tensor * 60
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")
    
    @staticmethod
    def interval_time2minutes(tensor, curr_unit):
        if curr_unit == 'ms':
            return torch.div(tensor, 1000 * 60, rounding_mode='trunc')
        elif curr_unit == 's':
            return torch.div(tensor, 60, rounding_mode='trunc')
        elif curr_unit == 'minute':
            return tensor
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")


    @classmethod
    def construct_df2dict(cls, cfg, df: pd.DataFrame, maxlen=None):
        if df is None: return None

        tmp_df = df[['stu_id','exer_id','label', 'timestamp', 'answer_time']].groupby('stu_id').agg(
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

        anwser_time_seq, _, _, = PadSeqUtil.pad_sequence(
            tmp_df['answer_time'].to_list(), dtype=np.int64,
            maxlen=maxlen
        )

        stu_id = tmp_df['stu_id'].to_numpy()[idx]
        
        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'label_seq': label_seq,
            'mask_seq': mask_seq,
            'time_seq': time_seq,
            'answer_time_seq': anwser_time_seq
        }
