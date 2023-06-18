# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import logging
from edustudio.utils.common import UnifyConfig
from edustudio.utils.callback import CallbackList


class BaseEvalFmt(object):
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg: UnifyConfig = cfg
        self.datafmt_cfg: UnifyConfig = cfg.datafmt_cfg
        self.evalfmt_cfg: UnifyConfig = cfg.evalfmt_cfg
        self.trainfmt_cfg: UnifyConfig = cfg.trainfmt_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger

    @classmethod
    def get_default_cfg(cls):
        parent_class = cls.__base__
        cfg = UnifyConfig(cls.default_cfg)
        if hasattr(parent_class, 'get_default_cfg'):
            cfg.update(parent_class.get_default_cfg(), update_unknown_key_only=True)
        return cfg

    def eval(self, **kwargs):
        pass
    
    def _check_params(self):
        pass
    
    def set_callback_list(self, callbacklist: CallbackList):
        self.callback_list = callbacklist

    def set_dataloaders(self, train_loader, test_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def add_extra_data(self, **kwargs):
        self.extra_data = kwargs

