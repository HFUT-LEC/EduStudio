from edustudio.utils.common import UnifyConfig
import logging


class BaseMid2Cache(object):
    default_cfg = {}

    def __init__(self, m2c_cfg) -> None:
        self.logger = logging.getLogger("edustudio")
        self.m2c_cfg = m2c_cfg
        self._check_params()

    def _check_params(self):
        pass

    @classmethod
    def from_cfg(cls, cfg: UnifyConfig):
        return cls(cfg.datatpl_cfg.get(cls.__name__))
    
    @classmethod
    def get_default_cfg(cls, **kwargs):
        cfg = UnifyConfig()
        for _cls in cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg
    
    def process(self, **kwargs):
        pass

    def set_dt_info(self, dt_info, **kwargs):
        pass
