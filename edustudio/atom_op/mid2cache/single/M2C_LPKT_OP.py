from ..common import BaseMid2Cache
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class M2C_LPKT_OP(BaseMid2Cache):
    default_cfg = {
        'answer_time_unit': 'ms', # ['ms', 's', 'minute']
        'start_timestamp_unit': 'ms', # ['ms', 's', 'minute']
    }

    def process(self, **kwargs):
        df_train_folds = kwargs['df_train_folds']
        df_valid_folds = kwargs['df_valid_folds']
        df_test_folds = kwargs['df_test_folds']

        self.dt_info = kwargs['dt_info']
        self.dt_info['answer_time_count_list'] = []
        self.dt_info['interval_time_count_list'] = []

        for idx, (train_dict, test_dict) in enumerate(zip(df_train_folds, df_test_folds)):
            self.train_dict = train_dict
            self.test_dict = test_dict
            if df_valid_folds is not None and len(df_valid_folds) > 0:
                self.val_dict = df_valid_folds[idx]
                
            self.construct_answer_time()
            self.construct_interval_time()

        return kwargs

    def construct_answer_time(self):
        # 1. train, val, test中所有answer_time变成秒
        curr_unit = self.m2c_cfg['answer_time_unit']
        answer_time = []
        answer_time.append(self.answer_time2seconds(self.train_dict['answer_time_seq:float_seq'], curr_unit))
        self.train_dict['answer_time_seq:float_seq'] = answer_time[-1]
        if self.val_dict is not None:
            answer_time.append(self.answer_time2seconds(self.val_dict['answer_time_seq:float_seq'], curr_unit))
            self.val_dict['answer_time_seq:float_seq'] = answer_time[-1]
        answer_time.append(self.answer_time2seconds(self.test_dict['answer_time_seq:float_seq'], curr_unit))
        self.test_dict['answer_time_seq:float_seq'] = answer_time[-1]

        # 2. 离散化
        at_lbe = LabelEncoder()
        at_lbe.fit(np.concatenate(answer_time).flatten())
        self.train_dict['answer_time_seq:float_seq'] = at_lbe.transform(
            self.train_dict['answer_time_seq:float_seq'].flatten()).reshape(self.train_dict['answer_time_seq:float_seq'].shape)
        if self.val_dict is not None:
            self.val_dict['answer_time_seq:float_seq'] = at_lbe.transform(
                self.val_dict['answer_time_seq:float_seq'].flatten()).reshape(self.val_dict['answer_time_seq:float_seq'].shape)
        self.test_dict['answer_time_seq:float_seq'] = at_lbe.transform(
            self.test_dict['answer_time_seq:float_seq'].flatten()).reshape(self.test_dict['answer_time_seq:float_seq'].shape)

        self.dt_info['answer_time_count_list'].append(len(at_lbe.classes_))
        

    def construct_interval_time(self):
        # 1. 计算train, val, test中的interval_time, 将interval time变成分钟
        curr_unit = self.m2c_cfg['start_timestamp_unit']
        interval_time = []
        train_it = self.train_dict['start_timestamp_seq:float_seq'] - np.concatenate([self.train_dict['start_timestamp_seq:float_seq'][:,[0]], self.train_dict['start_timestamp_seq:float_seq'][:,0:-1]], axis=1)
        train_it[train_it < 0] = 0
        train_it = self.interval_time2minutes(train_it, curr_unit)
        train_it[train_it > 43200] = 43200
        interval_time.append(train_it)
        test_it = self.test_dict['start_timestamp_seq:float_seq'] - np.concatenate([self.test_dict['start_timestamp_seq:float_seq'][:,[0]], self.test_dict['start_timestamp_seq:float_seq'][:,0:-1]], axis=1)
        test_it[test_it < 0] = 0
        test_it = self.interval_time2minutes(test_it, curr_unit)
        test_it[test_it > 43200] = 43200
        interval_time.append(test_it)
        if self.val_dict is not None:
            val_it = self.val_dict['start_timestamp_seq:float_seq'] - np.concatenate([self.val_dict['start_timestamp_seq:float_seq'][:,[0]], self.val_dict['start_timestamp_seq:float_seq'][:,0:-1]], axis=1)
            val_it[val_it < 0] = 0
            val_it = self.interval_time2minutes(val_it, curr_unit)
            val_it[val_it > 43200] = 43200
            interval_time.append(val_it)
        
        # 2. 离散化
        it_lbe = LabelEncoder()
        it_lbe.fit(np.concatenate(interval_time).flatten())
        self.train_dict['interval_time_seq'] = it_lbe.transform(train_it.flatten()).reshape(train_it.shape)
        if self.val_dict is not None:
            self.val_dict['interval_time_seq'] = it_lbe.transform(val_it.flatten()).reshape(val_it.shape)
        self.test_dict['interval_time_seq'] = it_lbe.transform(test_it.flatten()).reshape(test_it.shape)

        self.dt_info['interval_time_count_list'].append(len(it_lbe.classes_))

    @staticmethod
    def answer_time2seconds(tensor, curr_unit):
        if curr_unit == 'ms':
            return tensor // 1000
        elif curr_unit == 's':
            return tensor
        elif curr_unit == 'minute':
            return tensor * 60
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")
    
    @staticmethod
    def interval_time2minutes(tensor, curr_unit):
        if curr_unit == 'ms':
            return tensor // (1000 * 60)
        elif curr_unit == 's':
            return tensor // 60
        elif curr_unit == 'minute':
            return tensor
        else:
            ValueError(f"unsupported answer_time_unit: {curr_unit}")
