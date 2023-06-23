from edustudio.utils.common import UnifyConfig
import logging
import os


class BaseRaw2Mid(object):
    def __init__(self, dt, rawpath, midpath) -> None:
        self.dt = dt
        self.rawpath = rawpath
        self.midpath = midpath
        self.logger = logging.getLogger("edustudio")
        if not os.path.exists(self.midpath):
            os.makedirs(self.midpath)

    @classmethod
    def from_cfg(cls, cfg: UnifyConfig):
        rawdata_folder_path = f"{cfg.frame_cfg.data_folder_path}/rawdata"
        middata_folder_path = f"{cfg.frame_cfg.data_folder_path}/middata"
        dt = cfg.dataset
        return cls(dt, rawdata_folder_path, middata_folder_path)
    
    def process(self, **kwargs):
        self.logger.info(f"{self.__class__.__name__} start !")
    
    @classmethod
    def from_cli(cls, dt, rawpath="./rawpath", midpath="./midpath"):
        obj = cls(dt, rawpath, midpath)
        obj.process()

# __cli_api_dict__ = {
#     'BaseRaw2Mid': BaseRaw2Mid.from_cli
# }
