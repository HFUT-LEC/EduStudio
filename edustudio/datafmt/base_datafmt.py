import logging
from edustudio.utils.common import UnifyConfig
from .utils.common import BigfileDownloader, DecompressionUtil
import yaml
import re
import os
from torch.utils.data import Dataset
import importlib

class BaseDataFmt(Dataset):
    default_cfg = {'seed': 2023}

    def __init__(self, cfg):
        self.cfg: UnifyConfig = cfg
        self.datafmt_cfg: UnifyConfig = cfg.datafmt_cfg
        self.evalfmt_cfg: UnifyConfig = cfg.evalfmt_cfg
        self.trainfmt_cfg: UnifyConfig = cfg.trainfmt_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger
        self._check_params()
        self._init_data_before_dt_info()
        self.datafmt_cfg['dt_info'] = {}
        self._stat_dataset_info()
        self._init_data_after_dt_info()
        self.logger.info(self.datafmt_cfg['dt_info'])

    @classmethod
    def get_default_cfg(cls, **kwargs):
        cfg = UnifyConfig()
        for _cls in cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg)
    
    def _check_params(self):
        pass

    def _stat_dataset_info(self):
        pass

    def _init_data_before_dt_info(self):
        pass

    def _init_data_after_dt_info(self):
        pass

    def get_extra_data(self):
        return {}
    
    @classmethod
    def download_dataset(cls, cfg):
        dt_name = cfg.dataset
        cfg.logger.warning(f"Can't find dataset files of {dt_name} in local environment!")
        cfg.logger.info(f"Prepare to download {dt_name} from Internet.")
        fph = cfg.frame_cfg['DT_INFO_FILE_PATH']
        dataset_info = cls.read_yml_file(fph)
        if dt_name not in dataset_info:
            raise Exception("Can't find dataset files from Local and Internet!")

        fph_tmp = f"{cfg.frame_cfg.DATA_FOLDER_PATH}/{dt_name}.zip.tmp"
        fph_zip = f"{cfg.frame_cfg.DATA_FOLDER_PATH}/{dt_name}.zip"
        if not os.path.exists(fph_zip):
            if os.path.exists(fph_tmp):
                os.remove(fph_tmp)
            BigfileDownloader.download(
                url=dataset_info[dt_name]['url'], title=f"{dt_name} downloading...", 
                filepath=fph_tmp
            )
            os.rename(fph_tmp, fph_zip)
        else:
            cfg.logger.info(f"Find a zip file of {dt_name} at {fph_zip}, skip downloading process")

        DecompressionUtil.unzip_file(
            zip_src=fph_zip, dst_dir=cfg.frame_cfg.DATA_FOLDER_PATH
        )
        
    @classmethod
    def read_yml_file(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=cls._build_yaml_loader())
        return config

    @staticmethod
    def _build_yaml_loader():
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def build_datasets(self):
        return None, None, None



# class BaseProxyDataFmt(object):
#     default_cfg = {'backbone_datafmt_cls': 'BaseDataFmt'}

#     @classmethod
#     def from_cfg(cls, cfg):
#         backbone_datafmt_cls = cls.get_backbone_cls(cfg.datafmt_cfg.backbone_datafmt_cls)
#         cls.regiter_functions(backbone_cls=backbone_datafmt_cls)
#         return backbone_datafmt_cls.from_cfg(cfg)
    
#     @classmethod
#     def regiter_functions(cls, backbone_cls):
#         pass

#     @classmethod
#     def get_backbone_cls(cls, backbone_datafmt_cls):
#         if isinstance(backbone_datafmt_cls, str):
#             backbone_datafmt_cls = importlib.import_module('edustudio.datafmt').\
#                 __getattribute__(backbone_datafmt_cls)
#         elif isinstance(backbone_datafmt_cls, BaseDataFmt):
#             backbone_datafmt_cls = backbone_datafmt_cls
#         else:
#             raise ValueError(f"Unknown type of backbone_datafmt_cls: {backbone_datafmt_cls}")
#         return backbone_datafmt_cls

#     @classmethod
#     def get_default_cfg(cls, backbone_datafmt_cls):
#         parent_class = cls.__base__
#         cfg = UnifyConfig(cls.default_cfg)
#         cfg.backbone_datafmt_cls = backbone_datafmt_cls or cfg.backbone_datafmt_cls
#         if hasattr(parent_class, 'get_default_cfg'):
#             cfg.update(parent_class.get_default_cfg(cfg.backbone_datafmt_cls), update_unknown_key_only=True)
#         if cls is BaseProxyDataFmt:
#             backbone_datafmt_cls = cls.get_backbone_cls(cfg.backbone_datafmt_cls)
#             cfg.update(backbone_datafmt_cls.get_default_cfg(), update_unknown_key_only=True)
#         return cfg


class BaseProxyDataFmt(object):
    default_cfg = {'backbone_datafmt_cls': 'BaseDataFmt'}

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
        elif issubclass(backbone_datafmt_cls, BaseDataFmt):
            backbone_datafmt_cls = backbone_datafmt_cls
        else:
            raise ValueError(f"Unknown type of backbone_datafmt_cls: {backbone_datafmt_cls}")
        return backbone_datafmt_cls
    
    @classmethod
    def get_new_cls(cls, p_cls):
        new_cls = type(cls.__name__ + "_proxy", (cls, p_cls), {})
        return new_cls

    @classmethod
    def get_default_cfg(cls, backbone_datafmt_cls):
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
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg
