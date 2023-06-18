# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import torch.nn as nn
import logging
from edustudio.utils.common import UnifyConfig
import importlib

class BaseModel(nn.Module):
    default_cfg = {}
    def __init__(self, cfg):
        super().__init__()
        self.cfg: UnifyConfig = cfg
        self.datafmt_cfg: UnifyConfig = cfg.datafmt_cfg
        self.evalfmt_cfg: UnifyConfig = cfg.evalfmt_cfg
        self.trainfmt_cfg: UnifyConfig = cfg.trainfmt_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg)
    
    def add_extra_data(self, **kwargs):
        pass

    @classmethod
    def get_default_cfg(cls, **kwargs):
        cfg = UnifyConfig()
        for _cls in cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg


# class BaseProxyModel(object):
#     default_cfg = {
#         'backbone_model_cls': 'BaseModel'
#     }
    
#     @classmethod
#     def from_cfg(cls, cfg):
#         backbone_model_cls = cls.get_backbone_cls(cfg.model_cfg.backbone_model_cls)
#         cls.register_functions(backbone_cls=backbone_model_cls)
#         return backbone_model_cls.from_cfg(cfg)
    
#     @classmethod
#     def register_functions(cls, backbone_cls):
#         pass

#     @classmethod
#     def get_backbone_cls(cls, backbone_model_cls):
#         if isinstance(backbone_model_cls, str):
#             backbone_model_cls = importlib.import_module('edustudio.model').\
#                 __getattribute__(backbone_model_cls)
#         elif isinstance(backbone_model_cls, BaseModel):
#             backbone_model_cls = backbone_model_cls
#         else:
#             raise ValueError(f"Unknown type of backbone_model_cls: {backbone_model_cls}")
#         return backbone_model_cls

#     @classmethod
#     def get_default_cfg(cls, backbone_model_cls, **kwargs):
#         parent_class = cls.__base__
#         cfg = UnifyConfig(cls.default_cfg)
#         cfg.backbone_model_cls = backbone_model_cls or cfg.backbone_model_cls
#         if hasattr(parent_class, 'get_default_cfg'):
#             cfg.update(parent_class.get_default_cfg(backbone_model_cls=cfg.backbone_model_cls, **kwargs), update_unknown_key_only=True)
#         if cls is BaseProxyModel:
#             backbone_model_cls = cls.get_backbone_cls(backbone_model_cls=cfg.backbone_model_cls, **kwargs)
#             cfg.update(backbone_model_cls.get_default_cfg(**kwargs), update_unknown_key_only=True)
#         return cfg



class BaseProxyModel(object):
    default_cfg = {
        'backbone_model_cls': 'BaseModel'
    }
    
    @classmethod
    def from_cfg_proxy(cls, cfg):
        backbone_model_cls = cls.get_backbone_cls(cfg.model_cfg.backbone_model_cls)
        new_cls = cls.get_new_cls(p_cls=backbone_model_cls)
        return new_cls.from_cfg(cfg)
    
    @classmethod
    def get_backbone_cls(cls, backbone_model_cls):
        if isinstance(backbone_model_cls, str):
            backbone_model_cls = importlib.import_module('edustudio.model').\
                __getattribute__(backbone_model_cls)
        elif issubclass(backbone_model_cls, BaseModel):
            backbone_model_cls = backbone_model_cls
        else:
            raise ValueError(f"Unknown type of backbone_model_cls: {backbone_model_cls}")
        return backbone_model_cls

    @classmethod
    def get_new_cls(cls, p_cls):
        new_cls = type(cls.__name__ + "_proxy", (cls, p_cls), {})
        return new_cls

    @classmethod
    def get_default_cfg(cls, backbone_model_cls):
        bb_cls = None
        if backbone_model_cls is not None:
            bb_cls = cls.get_backbone_cls(backbone_model_cls)
        else:
            for _cls in cls.__mro__:
                if not hasattr(_cls, 'default_cfg'):
                    break
                bb_cls = _cls.default_cfg.get('backbone_model_cls', None)
                if bb_cls is not None: break
            assert bb_cls is not None
            bb_cls = cls.get_backbone_cls(bb_cls)
        
        cfg = UnifyConfig()
        cfg.backbone_model_cls = bb_cls
        cfg.backbone_model_cls_name = bb_cls.__name__
        new_cls = cls.get_new_cls(p_cls=bb_cls)
        for _cls in new_cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg
