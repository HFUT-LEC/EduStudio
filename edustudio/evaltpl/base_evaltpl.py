# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import logging
from edustudio.utils.common import UnifyConfig
from edustudio.utils.callback import CallbackList


class BaseEvalTPL(object):
    """The baisc protocol for implementing a evaluate template
    """
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg: UnifyConfig = cfg
        self.datatpl_cfg: UnifyConfig = cfg.datatpl_cfg
        self.evaltpl_cfg: UnifyConfig = cfg.evaltpl_cfg
        self.traintpl_cfg: UnifyConfig = cfg.traintpl_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.modeltpl_cfg: UnifyConfig = cfg.modeltpl_cfg
        self.logger: logging.Logger = logging.getLogger("edustudio")
        self.name = self.name = self.__class__.__name__
        self._check_params()

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

    def set_dataloaders(self, train_loader, test_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def add_extra_data(self, **kwargs):
        self.extra_data = kwargs

