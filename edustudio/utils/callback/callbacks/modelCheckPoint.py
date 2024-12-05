# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from .callback import Callback
from typing import Tuple, List
import torch
import os
import shutil
import glob
from .baseLogger import BaseLogger
import numpy as np
from collections import namedtuple


class Metric(object):
    def __init__(self, name, type_):
        assert type_ in ['max', 'min']
        self.name = name
        self.type_ = type_
        self.best_epoch = 1
        self.best_log = dict()

        if self.type_ == 'max':
            self.best_value = -np.inf
        else:
            self.best_value = np.inf

    def better_than(self, value):
        return value < self.best_value if self.type_ == 'max' else value > self.best_value

    def update(self, epoch, value, log):
        self.best_epoch = epoch
        self.best_value = value
        self.best_log = log

class ModelCheckPoint(Callback):
    def __init__(self, metric_list: List[list], save_folder_path, save_best_only=True):
        super().__init__()
        self.save_folder_path = save_folder_path
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)
        self.save_best_only = save_best_only
        self.metric_list = [Metric(name=metric_name, type_=metric_type) for metric_name, metric_type in metric_list]
        

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        for metric in self.metric_list:
            if not metric.better_than(logs[metric.name]):
                metric.update(epoch=epoch, value=logs[metric.name], log=logs)
        
                self.logger.info(f"Update best epoch as {[epoch]} for {metric.name}!")
                filepath = os.path.join(self.save_folder_path, f"best-epoch-for-{metric.name}.pth")
                torch.save(obj=self.model.state_dict(), f=filepath)

        if not self.save_best_only:
            filepath = os.path.join(self.save_folder_path, f"{epoch:03d}.pth")
            torch.save(obj=self.model.state_dict(), f=filepath)            

    def on_train_end(self, logs=None, **kwargs):
        self.logger.info("==="*10)
        for metric in self.metric_list:
            self.logger.info(f"[For {metric.name}], the Best Epoch is: {metric.best_epoch}, the value={metric.best_value:.4f}")
            for cb in self.callback_list.callbacks:
                if isinstance(cb, BaseLogger):
                    cb.on_epoch_end(metric.best_epoch, logs=metric.best_log)
            # rename best epoch filename
            os.renames(
                os.path.join(self.save_folder_path, f"best-epoch-for-{metric.name}.pth"), 
                os.path.join(self.save_folder_path, f"best-epoch-{metric.best_epoch:03d}-for-{metric.name}.pth")
            )
        self.logger.info("===" * 10)
