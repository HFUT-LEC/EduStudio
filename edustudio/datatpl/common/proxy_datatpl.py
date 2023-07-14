import importlib
from .base_datatpl import BaseDataTPL
from edustudio.utils.common import UnifyConfig


class BaseProxyDataTPL(object):
    """The basic protocol for implementing a proxy data template
    """
    default_cfg = {'backbone_datatpl_cls': 'BaseDataTPL'}

    @classmethod
    def from_cfg_proxy(cls, cfg):
        """an interface to instantiate a proxy model

        Args:
            cfg (UnifyConfig): the global config object

        Returns:
           BaseProxyDataTPL
        """
        backbone_datatpl_cls = cls.get_backbone_cls(cfg.datatpl_cfg.backbone_datatpl_cls)
        new_cls = cls.get_new_cls(p_cls=backbone_datatpl_cls)
        return new_cls.from_cfg(cfg)

    @classmethod
    def get_backbone_cls(cls, backbone_datatpl_cls):
        if isinstance(backbone_datatpl_cls, str):
            backbone_datatpl_cls = importlib.import_module('edustudio.datatpl').\
                __getattribute__(backbone_datatpl_cls)
        elif issubclass(backbone_datatpl_cls, BaseDataTPL):
            backbone_datatpl_cls = backbone_datatpl_cls
        else:
            raise ValueError(f"Unknown type of backbone_datatpl_cls: {backbone_datatpl_cls}")
        return backbone_datatpl_cls
    
    @classmethod
    def get_new_cls(cls, p_cls):
        """dynamic inheritance

        Args:
            p_cls (BaseModel): parent class

        Returns:
            BaseProxyModel: A inherited class
        """
        new_cls = type(cls.__name__ + "_proxy", (cls, p_cls), {})
        return new_cls

    @classmethod
    def get_default_cfg(cls, backbone_datatpl_cls, **kwargs):
        """Get the final default_cfg

        Args:
            backbone_datatpl_cls (BaseDataTPL): backbone data template class name

        Returns:
            UnifyConfig: the final default config object
        """
        bb_cls = None
        if backbone_datatpl_cls is not None:
            bb_cls = cls.get_backbone_cls(backbone_datatpl_cls)
        else:
            for _cls in cls.__mro__:
                if not hasattr(_cls, 'default_cfg'):
                    break
                bb_cls = _cls.default_cfg.get('backbone_datatpl_cls', None)
                if bb_cls is not None: break
            assert bb_cls is not None
            bb_cls = cls.get_backbone_cls(bb_cls)
        
        cfg = UnifyConfig()
        cfg.backbone_datatpl_cls = bb_cls
        cfg.backbone_datatpl_cls_name = bb_cls.__name__
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
