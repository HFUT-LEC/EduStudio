# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from .callback import Callback
from collections import defaultdict
from typing import List, Sequence, Optional
import json
import os
import numpy as np
from edustudio.utils.common import NumpyEncoder


class History(Callback):
    def __init__(self, folder_path, exclude_metrics=(), plot_curve=False):
        super(History, self).__init__()
        self.log_as_time = {}
        self.exclude_metrics = set(exclude_metrics)
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            
        self.plot_curve = plot_curve
        if self.plot_curve:
            import matplotlib
            if not os.path.exists(self.folder_path+"/plots/"):
                os.makedirs(self.folder_path+"/plots/")
        self.names = None
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        self.log_as_time[epoch] = {k:v for k,v in logs.items() if isinstance(v, (int, str, float)) and k not in self.exclude_metrics}
        if self.names is None:
            self.names = set(self.log_as_time[epoch].keys())
        else:
            assert self.names == set(self.log_as_time[epoch].keys())
        

    def on_train_end(self, logs=None, **kwargs):
        self.dump_json(self.log_as_time, os.path.join(self.folder_path, "history.json"))

        if self.plot_curve:
            self.logger.info("[CALLBACK]-History: Plot Curve...")
            self.plot()
            self.logger.info("[CALLBACK]-History: Plot Succeed!")

    def plot(self):
        import matplotlib.pyplot as plt
        epoch_num = len(self.log_as_time)
        x_arr = np.arange(1, epoch_num+1)
        for name in self.names:
            value_arr = [self.log_as_time[i][name] for i in range(1, epoch_num+1)]
            plt.figure()
            plt.title(name)
            plt.xlabel("epoch")
            plt.ylabel("value")
            plt.plot(x_arr, value_arr)
            plt.autoscale()
            plt.savefig(f"{self.folder_path}/plots/{name}.png", dpi=500, bbox_inches='tight', pad_inches=0.1)
        
    @staticmethod
    def dump_json(data, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf8') as f:
            json.dump(data, fp=f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
