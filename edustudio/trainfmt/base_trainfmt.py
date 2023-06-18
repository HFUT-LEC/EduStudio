# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import logging
import importlib
from edustudio.utils.common import UnifyConfig
from edustudio.datafmt import BaseDataFmt, BaseProxyDataFmt
from edustudio.model import BaseModel, BaseProxyModel
from edustudio.evalfmt import BaseEvalFmt
from typing import Sequence
from edustudio.utils.common import set_same_seeds


class BaseTrainFmt(object):
    default_cfg = {}

    def __init__(self, cfg: UnifyConfig):
        self.cfg: UnifyConfig = cfg
        self.datafmt_cfg: UnifyConfig = cfg.datafmt_cfg
        self.evalfmt_cfg: UnifyConfig = cfg.evalfmt_cfg
        self.trainfmt_cfg: UnifyConfig = cfg.trainfmt_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger

        if isinstance(self.datafmt_cfg['cls'], str):
            self.datafmt_cls: BaseDataFmt = importlib.import_module('edustudio.datafmt').\
                __getattribute__(self.datafmt_cfg['cls'])
        else:
            self.datafmt_cls = self.datafmt_cfg['cls']
        
        if isinstance(self.model_cfg['cls'], str):
            self.model_cls: BaseModel = importlib.import_module('edustudio.model').\
                __getattribute__(self.model_cfg['cls'])
        else:
            self.model_cls = self.model_cfg['cls']
        
        self.evalfmt_clses: Sequence[BaseEvalFmt] = [
            importlib.import_module('edustudio.evalfmt').__getattribute__(fmt) if isinstance(fmt, str) else fmt
            for fmt in self.evalfmt_cfg['clses'] 
        ]

        set_same_seeds(self.datafmt_cfg['seed'])
        if issubclass(self.datafmt_cls, BaseDataFmt):
            self.datafmt: BaseDataFmt = self.datafmt_cls.from_cfg(self.cfg)
        elif issubclass(self.datafmt_cls, BaseProxyDataFmt):
            self.datafmt: BaseDataFmt = self.datafmt_cls.from_cfg_proxy(self.cfg)
        else:
            raise ValueError(f"unsupported datafmt_cls: {self.datafmt_cls}")

        if issubclass(self.model_cls, BaseModel):
            self.model: BaseModel = self.model_cls.from_cfg(self.cfg)
        elif issubclass(self.model_cls, BaseProxyModel):
            self.model: BaseModel = self.model_cls.from_cfg_proxy(self.cfg)
        else:
            raise ValueError(f"unsupported model_cls: {self.model_cls}")
        
        self.evalfmts: Sequence[BaseEvalFmt] = [cls(self.cfg) for cls in self.evalfmt_clses]
        self._check_params()

    @classmethod
    def get_default_cfg(cls):
        parent_class = cls.__base__
        cfg = UnifyConfig(cls.default_cfg)
        if hasattr(parent_class, 'get_default_cfg'):
            cfg.update(parent_class.get_default_cfg(), update_unknown_key_only=True)
        return cfg
    
    def start(self):
        self.logger.info(f"TrainFmt {self.__class__.__base__} Started!")

    def _check_params(self):
        pass
