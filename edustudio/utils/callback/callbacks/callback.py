# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import torch.nn as nn
from ..modeState import ModeState
import logging


class Callback(object):
    def __init__(self):
        self.model = None
        self.mode_state = None
        self.logger = None
        self.callback_list = None

    def set_model(self, model: nn.Module):
        self.model = model

    def set_state(self, mode_state: ModeState):
        self.mode_state = mode_state

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def set_callback_list(self, callback_list):
        self.callback_list = callback_list

    def on_train_begin(self, logs=None, **kwargs):
        self.logger.info(f"[CALLBACK]-{self.__class__.__name__} has been registered!")

    def on_train_end(self, logs=None, **kwargs):
        pass

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        pass

    def on_train_batch_begin(self, batch, logs=None, **kwargs):
        pass

    def on_train_batch_end(self, batch, logs=None, **kwargs):
        pass
