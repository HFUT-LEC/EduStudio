# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from .callbacks.callback import Callback
from .callbacks.history import History
from .callbacks.baseLogger import BaseLogger
from typing import Sequence
import torch.nn as nn
from functools import reduce
from .modeState import ModeState
import logging


class CallbackList(object):
    def __init__(self, callbacks: Sequence[Callback], model: nn.Module, 
                 add_logger: bool = True, logger: logging.Logger = None):
        """

        :param callbacks:
        :param model:
        :param add_history:
        :param add_logger:
        """
        self.callbacks = list(filter(lambda x: isinstance(x, Callback), callbacks))
        self._add_default_callbacks(add_logger)
        self._reg_logger(logger or logging.getLogger())
        self._reg_model(model)
        self._reg_callback_list()
        self._set_mode_state(ModeState.START)

        self.curr_epoch = None

    def _reg_model(self, model: nn.Module):
        self.model = model
        for cb in self.callbacks:
            cb.set_model(model)

    def _add_default_callbacks(self, add_logger: bool):
        self._logger = reduce(
            lambda x, y: x or y, map(lambda x: x if isinstance(x, BaseLogger) else None, self.callbacks)
        )
        if add_logger and not self._logger:
            self._logger = BaseLogger()
            self.callbacks.append(self._logger)

    def _set_mode_state(self, mode_state: ModeState):
        self.mode_state = mode_state
        for cb in self.callbacks:
            cb.set_state(mode_state=mode_state)

    def _reg_logger(self, logger: logging.Logger):
        self.logger = logger
        for cb in self.callbacks:
            cb.set_logger(logger)

    def _reg_callback_list(self):
        for cb in self.callbacks:
            cb.set_callback_list(self)

    def on_train_begin(self, logs=None, **kwargs):
        assert self.mode_state == ModeState.START
        self.curr_epoch = 1
        self._set_mode_state(ModeState.TRAINING)
        for cb in self.callbacks:
            cb.on_train_begin(logs, **kwargs)

    def on_train_end(self, logs=None, **kwargs):
        assert self.mode_state == ModeState.TRAINING
        self.curr_epoch = None
        self._set_mode_state(ModeState.END)
        for cb in self.callbacks:
            cb.on_train_end(logs, **kwargs)

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        assert self.mode_state == ModeState.TRAINING
        assert epoch == self.curr_epoch
        self.model.train()
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs, **kwargs)

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        assert self.mode_state == ModeState.TRAINING
        assert epoch == self.curr_epoch
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs, **kwargs)
        self.curr_epoch += 1

    def on_train_batch_begin(self, batch, logs=None, **kwargs):
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs, **kwargs)

    def on_train_batch_end(self, batch, logs=None, **kwargs):
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs, **kwargs)
