# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from typing import List
from .callback import Callback
import numpy as np


class Metric(object):
    def __init__(self, name, type_):
        assert type_ in ['max', 'min']
        self.name = name
        self.type_ = type_
        self.best_epoch = 1
        if self.type_ == 'max':
            self.best_value = -np.inf
        else:
            self.best_value = np.inf

    def better_than(self, value):
        return value < self.best_value if self.type_ == 'max' else value > self.best_value

    def update(self, epoch, value):
        self.best_epoch = epoch
        self.best_value = value


class EarlyStopping(Callback):
    def __init__(self, metric_list:List[list], num_stop_rounds: int = 20, start_round=1):
        """_summary_

        Args:
            metric_list (List[list]): [['rmse', 'min'],['ndcg', 'max']]
            num_stop_rounds (int, optional): all metrics have no improvement in latest num_stop_rounds, suggest to stop training. Defaults to 20.
            start_round (int, optional): start detecting from epoch start_round, . Defaults to 1.
        """
        super().__init__()
        assert num_stop_rounds >= 1
        assert start_round >= 1
        self.start_round = start_round
        self.num_stop_round = num_stop_rounds
        self.stop_training = False
        self.metric_list = [Metric(name=metric_name, type_=metric_type) for metric_name, metric_type in metric_list]

    def on_train_begin(self, logs=None, **kwargs):
        super().on_train_begin()
        self.model.share_callback_dict['stop_training'] = False

    def on_epoch_end(self, epoch: int, logs: dict = None, **kwargs):
        flag = True
        for metric in self.metric_list:
            if not metric.better_than(logs[metric.name]):
                metric.update(epoch=epoch, value=logs[metric.name])
        
            if self.start_round <= epoch:
                if epoch - metric.best_epoch < self.num_stop_round:
                    flag &= False

        if flag is True:
            self.logger.info("Suggest to stop training now")
            self.stop_training = True
            self.model.share_callback_dict['stop_training'] = True
