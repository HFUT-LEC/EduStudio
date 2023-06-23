import importlib
from .base_datatpl import BaseDataTPL
from edustudio.utils.common import UnifyConfig


class BaseProxyDataTPL(object):
    default_cfg = {'backbone_datafmt_cls': 'BaseDataTPL'}

    @classmethod
    def from_cfg_proxy(cls, cfg):
        backbone_datafmt_cls = cls.get_backbone_cls(cfg.datafmt_cfg.backbone_datafmt_cls)
        new_cls = cls.get_new_cls(p_cls=backbone_datafmt_cls)
        return new_cls.from_cfg(cfg)

    @classmethod
    def get_backbone_cls(cls, backbone_datafmt_cls):
        if isinstance(backbone_datafmt_cls, str):
            backbone_datafmt_cls = importlib.import_module('edustudio.datafmt').\
                __getattribute__(backbone_datafmt_cls)
        elif issubclass(backbone_datafmt_cls, BaseDataTPL):
            backbone_datafmt_cls = backbone_datafmt_cls
        else:
            raise ValueError(f"Unknown type of backbone_datafmt_cls: {backbone_datafmt_cls}")
        return backbone_datafmt_cls
    
    @classmethod
    def get_new_cls(cls, p_cls):
        new_cls = type(cls.__name__ + "_proxy", (cls, p_cls), {})
        return new_cls

    @classmethod
    def get_default_cfg(cls, backbone_datafmt_cls, **kwargs):
        bb_cls = None
        if backbone_datafmt_cls is not None:
            bb_cls = cls.get_backbone_cls(backbone_datafmt_cls)
        else:
            for _cls in cls.__mro__:
                if not hasattr(_cls, 'default_cfg'):
                    break
                bb_cls = _cls.default_cfg.get('backbone_datafmt_cls', None)
                if bb_cls is not None: break
            assert bb_cls is not None
            bb_cls = cls.get_backbone_cls(bb_cls)
        
        cfg = UnifyConfig()
        cfg.backbone_datafmt_cls = bb_cls
        cfg.backbone_datafmt_cls_name = bb_cls.__name__
        new_cls = cls.get_new_cls(p_cls=bb_cls)
        for _cls in new_cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            if issubclass(_cls, BaseProxyDataTPL):
                cfg.update(_cls.default_cfg, update_unknown_key_only=True)
            else:
                cfg.update(_cls.get_default_cfg(**kwargs), update_unknown_key_only=True)
                break
        return cfg
