import logging
from edustudio.utils.common import UnifyConfig
from torch.utils.data import Dataset
import copy
from ..utils.common import BigfileDownloader, DecompressionUtil
import yaml
import re
import os
import requests


class BaseDataTPL(Dataset):
    """The basic data protocol for implementing a data template
    """
    default_cfg = {'seed': 2023}

    def __init__(self, cfg):
        self.cfg: UnifyConfig = cfg
        self.datatpl_cfg: UnifyConfig = cfg.datatpl_cfg
        self.evaltpl_cfg: UnifyConfig = cfg.evaltpl_cfg
        self.traintpl_cfg: UnifyConfig = cfg.traintpl_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.modeltpl_cfg: UnifyConfig = cfg.modeltpl_cfg
        self.logger: logging.Logger = logging.getLogger("edustudio")
        self._check_params()

    @classmethod
    def get_default_cfg(cls, **kwargs):
        """Get the final default config object

        Returns:
            UnifyConfig: the final default config object
        """
        cfg = UnifyConfig()
        for _cls in cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        return cfg

    @classmethod
    def from_cfg(cls, cfg):
        """Instantiate data template

        Args:
            cfg (UnifyConfig): the global config object

        Returns:
            BaseDataTPL: instance of data template
        """
        return cls(cfg)
    
    def get_extra_data(self, **kwargs):
        """an interface to construct extra data except the data from forward API
        """
        return {}

    def _check_params(self):
        """check validation of default config
        """
        pass

    def _copy(self):
        """copy current instance
        """
        obj = copy.copy(self)
        return obj

    @classmethod
    def download_dataset(cls, cfg):
        """Download Dataset from the Internet

        Args:
            cfg (UnifyConfig):the global config object
        """
        dt_name = cfg.dataset
        cfg.logger.warning(f"Can't find dataset files of {dt_name} in local disk")
        
        fph = cfg.frame_cfg['DT_INFO_FILE_PATH']
        dataset_info = cls.read_yml_file(fph)
        dataset_info_from_cfg: dict = cfg['frame_cfg']['DT_INFO_DICT']
        dataset_info.update(dataset_info_from_cfg)

        if dt_name not in dataset_info:
            cfg.logger.info(f"Prepare download external datasets.yaml to find dataset:{dt_name}")
            url = "https://huggingface.co/datasets/lmcRS/edustudio-datasets/raw/main/datasets.yaml"
            cfg.logger.info(f"Eexternal datasets.yaml url: {url}")
            resp = requests.get(url)
            dataset_info_external = yaml.load(resp.text, Loader=cls._build_yaml_loader())
            if dt_name not in dataset_info_external:
                raise Exception("Can't find dataset files from local disk and online")
            else:
                dataset_info.update(dataset_info_external)

        cfg.logger.info(f"Prepare to download {dt_name} dataset from online")
        cfg.logger.info(f"Download_url: {dataset_info[dt_name]['middata_url']}")

        if not os.path.exists(cfg.frame_cfg.data_folder_path):
            os.makedirs(cfg.frame_cfg.data_folder_path)
        fph_tmp = f"{cfg.frame_cfg.data_folder_path}/{dt_name}-middata.zip.tmp"
        fph_zip = f"{cfg.frame_cfg.data_folder_path}/{dt_name}-middata.zip"
        if not os.path.exists(fph_zip):
            if os.path.exists(fph_tmp):
                os.remove(fph_tmp)
            BigfileDownloader.download(
                url=dataset_info[dt_name]['middata_url'], title=f"{dt_name} downloading...", 
                filepath=fph_tmp
            )
            os.rename(fph_tmp, fph_zip)
        else:
            cfg.logger.info(f"Find a zip file of {dt_name} at {fph_zip}, skip downloading process")

        DecompressionUtil.unzip_file(
            zip_src=fph_zip, dst_dir=cfg.frame_cfg.data_folder_path
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
