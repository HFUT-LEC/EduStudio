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
        self.datatpl_cfg: UnifyConfig = cfg.datatpl_cfg
        self.evaltpl_cfg: UnifyConfig = cfg.evaltpl_cfg
        self.traintpl_cfg: UnifyConfig = cfg.traintpl_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.modeltpl_cfg: UnifyConfig = cfg.modeltpl_cfg
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
#         'backbone_modeltpl_cls': 'BaseModel'
#     }
    
#     @classmethod
#     def from_cfg(cls, cfg):
#         backbone_modeltpl_cls = cls.get_backbone_cls(cfg.modeltpl_cfg.backbone_modeltpl_cls)
#         cls.register_functions(backbone_cls=backbone_modeltpl_cls)
#         return backbone_modeltpl_cls.from_cfg(cfg)
    
#     @classmethod
#     def register_functions(cls, backbone_cls):
#         pass

#     @classmethod
#     def get_backbone_cls(cls, backbone_modeltpl_cls):
#         if isinstance(backbone_modeltpl_cls, str):
#             backbone_modeltpl_cls = importlib.import_module('edustudio.model').\
#                 __getattribute__(backbone_modeltpl_cls)
#         elif isinstance(backbone_modeltpl_cls, BaseModel):
#             backbone_modeltpl_cls = backbone_modeltpl_cls
#         else:
#             raise ValueError(f"Unknown type of backbone_modeltpl_cls: {backbone_modeltpl_cls}")
#         return backbone_modeltpl_cls

#     @classmethod
#     def get_default_cfg(cls, backbone_modeltpl_cls, **kwargs):
#         parent_class = cls.__base__
#         cfg = UnifyConfig(cls.default_cfg)
#         cfg.backbone_modeltpl_cls = backbone_modeltpl_cls or cfg.backbone_modeltpl_cls
#         if hasattr(parent_class, 'get_default_cfg'):
#             cfg.update(parent_class.get_default_cfg(backbone_modeltpl_cls=cfg.backbone_modeltpl_cls, **kwargs), update_unknown_key_only=True)
#         if cls is BaseProxyModel:
#             backbone_modeltpl_cls = cls.get_backbone_cls(backbone_modeltpl_cls=cfg.backbone_modeltpl_cls, **kwargs)
#             cfg.update(backbone_modeltpl_cls.get_default_cfg(**kwargs), update_unknown_key_only=True)
#         return cfg



class BaseProxyModel(object):
    default_cfg = {
        'backbone_modeltpl_cls': 'BaseModel'
    }
    
    @classmethod
    def from_cfg_proxy(cls, cfg):
        backbone_modeltpl_cls = cls.get_backbone_cls(cfg.modeltpl_cfg.backbone_modeltpl_cls)
        new_cls = cls.get_new_cls(p_cls=backbone_modeltpl_cls)
        return new_cls.from_cfg(cfg)
    
    @classmethod
    def get_backbone_cls(cls, backbone_modeltpl_cls):
        if isinstance(backbone_modeltpl_cls, str):
            backbone_modeltpl_cls = importlib.import_module('edustudio.model').\
                __getattribute__(backbone_modeltpl_cls)
        elif issubclass(backbone_modeltpl_cls, BaseModel):
            backbone_modeltpl_cls = backbone_modeltpl_cls
        else:
            raise ValueError(f"Unknown type of backbone_modeltpl_cls: {backbone_modeltpl_cls}")
        return backbone_modeltpl_cls

    @classmethod
    def get_new_cls(cls, p_cls):
        new_cls = type(cls.__name__ + "_proxy", (cls, p_cls), {})
        return new_cls

    @classmethod
    def get_default_cfg(cls, backbone_modeltpl_cls):
        bb_cls = None
        if backbone_modeltpl_cls is not None:
            bb_cls = cls.get_backbone_cls(backbone_modeltpl_cls)
        else:
            for _cls in cls.__mro__:
                if not hasattr(_cls, 'default_cfg'):
                    break
                bb_cls = _cls.default_cfg.get('backbone_modeltpl_cls', None)
                if bb_cls is not None: break
            assert bb_cls is not None
            bb_cls = cls.get_backbone_cls(bb_cls)
        
        cfg = UnifyConfig()
        cfg.backbone_modeltpl_cls = bb_cls
        cfg.backbone_modeltpl_cls_name = bb_cls.__name__
        new_cls = cls.get_new_cls(p_cls=bb_cls)
        for _cls in new_cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg
