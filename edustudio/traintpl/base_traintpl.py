# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import logging
import importlib
from edustudio.utils.common import UnifyConfig
from edustudio.datatpl import BaseDataTPL, BaseProxyDataTPL
from edustudio.model import BaseModel, BaseProxyModel
from edustudio.evaltpl import BaseEvalTPL
from edustudio.datatpl import BaseDataTPL
from typing import Sequence
from edustudio.utils.common import set_same_seeds


class BaseTrainTPL(object):
    """The basic protocol for implementing a training template
    """

    default_cfg = {
        "seed": 2023
    }

    def __init__(self, cfg: UnifyConfig):
        self.cfg: UnifyConfig = cfg
        self.datatpl_cfg: UnifyConfig = cfg.datatpl_cfg
        self.evaltpl_cfg: UnifyConfig = cfg.evaltpl_cfg
        self.traintpl_cfg: UnifyConfig = cfg.traintpl_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.modeltpl_cfg: UnifyConfig = cfg.modeltpl_cfg
        self.logger: logging.Logger = logging.getLogger("edustudio")
        self._check_params()

        if isinstance(self.datatpl_cfg['cls'], str):
            self.datatpl_cls: BaseDataTPL = importlib.import_module('edustudio.datatpl').\
                __getattribute__(self.datatpl_cfg['cls'])
        else:
            self.datatpl_cls = self.datatpl_cfg['cls']
        
        if isinstance(self.modeltpl_cfg['cls'], str):
            self.model_cls: BaseModel = importlib.import_module('edustudio.model').\
                __getattribute__(self.modeltpl_cfg['cls'])
        else:
            self.model_cls = self.modeltpl_cfg['cls']
        
        self.evaltpl_clses: Sequence[BaseEvalTPL] = [
            importlib.import_module('edustudio.evaltpl').__getattribute__(tpl) if isinstance(tpl, str) else tpl
            for tpl in self.evaltpl_cfg['clses'] 
        ]

        set_same_seeds(self.datatpl_cfg['seed'])

        self.model = self.get_model_obj()
        self.datatpl = self.get_datatpl_obj()
        self.evaltpls = self.get_evaltpl_obj_list()

    @classmethod
    def get_default_cfg(cls):
        parent_class = cls.__base__
        cfg = UnifyConfig(cls.default_cfg)
        if hasattr(parent_class, 'get_default_cfg'):
            cfg.update(parent_class.get_default_cfg(), update_unknown_key_only=True)
        return cfg
    
    def start(self):
        """entrypoint of starting a training process
        """
        self.logger.info(f"TrainTPL {self.__class__} Started!")
        set_same_seeds(self.traintpl_cfg['seed'])

    def _check_params(self):
        """check validation of default config
        """
        pass

    def get_model_obj(self):
        if issubclass(self.model_cls, BaseModel):
            model: BaseModel = self.model_cls.from_cfg(self.cfg)
        elif issubclass(self.model_cls, BaseProxyModel):
            model: BaseModel = self.model_cls.from_cfg_proxy(self.cfg)
        else:
            raise ValueError(f"unsupported model_cls: {self.model_cls}")
        return model

    def get_datatpl_obj(self):
        if issubclass(self.datatpl_cls, BaseDataTPL):
            datatpl: BaseDataTPL = self.datatpl_cls.from_cfg(self.cfg)
        elif issubclass(self.datatpl_cls, BaseProxyDataTPL):
            datatpl: BaseDataTPL = self.datatpl_cls.from_cfg_proxy(self.cfg)
        else:
            raise ValueError(f"unsupported datatpl_cls: {self.datatpl_cls}")
        return datatpl

    def get_evaltpl_obj_list(self):
        evaltpls: Sequence[BaseEvalTPL] = [cls(self.cfg) for cls in self.evaltpl_clses]
        return evaltpls

