# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from .callback import Callback
from collections import defaultdict


class BaseLogger(Callback):
    def __init__(self, logger=None, **kwargs):
        super(BaseLogger, self).__init__()
        self.log = print
        if logger is not None and hasattr(logger, 'info'):
            self.log = logger.info

        self.join_str = kwargs.get("join_str", " | ")
        self.group_by_contains = kwargs.get('group_by_contains', ('loss'))
        self.group_by_count = kwargs.get('group_by_count', 5)
        assert self.group_by_count >= 1 and type(self.group_by_count) is int

    def on_train_begin(self, logs=None, **kwargs):
        super().on_train_begin()
        self.log("Start Training...")

    def on_epoch_end(self, epoch: int, logs: dict = None, **kwargs):
        info = f"[EPOCH={epoch:03d}]: "
        flag = [False] * len(logs)
        for group_rule in self.group_by_contains:
            v_list = []
            for i, (k, v) in enumerate(logs.items()):
                if group_rule in k and not flag[i]:
                    v_list.append(f"{k}: {v:.4f}")
                    flag[i] = True
            if len(v_list) > 0: self.log(f"{info}{self.join_str.join(v_list)}")
        
        logs = {k: logs[k] for i, k in enumerate(logs) if not flag[i]}

        v_list = []
        for i, (k, v) in enumerate(logs.items()):
            v_list.append(f"{k}: {v:.4f}")
            if (i+1) % self.group_by_count == 0:
                self.log(f"{info}{self.join_str.join(v_list)}")
                v_list = []
        if len(v_list) > 0: self.log(f"{info}{self.join_str.join(v_list)}")

    def on_train_end(self, logs=None, **kwargs):
        self.log("Training Completed!")
