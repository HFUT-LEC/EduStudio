# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from edustudio.utils.common import UnifyConfig, PathUtil, Logger
import os


def init_all(cfg: UnifyConfig):
    frame_cfg = cfg.frame_cfg
    dataset = cfg.dataset
    trainfmt_cls =cfg.trainfmt_cfg.cls if isinstance(cfg.trainfmt_cfg.cls, str) else cfg.trainfmt_cfg.cls.__name__

    frame_cfg.data_folder_path = f"{frame_cfg.DATA_FOLDER_PATH}/{dataset}"
    # PathUtil.check_path_exist(frame_cfg.data_folder_path)

    frame_cfg.TEMP_FOLDER_PATH = os.path.realpath(frame_cfg.TEMP_FOLDER_PATH)
    frame_cfg.ARCHIVE_FOLDER_PATH = os.path.realpath(frame_cfg.ARCHIVE_FOLDER_PATH)

    frame_cfg.temp_folder_path = f"{frame_cfg.TEMP_FOLDER_PATH}/{dataset}/{trainfmt_cls}/{cfg.model_cfg.cls}/{frame_cfg.ID}"
    frame_cfg.archive_folder_path = f"{frame_cfg.ARCHIVE_FOLDER_PATH}/{dataset}/{trainfmt_cls}/{cfg.model_cfg.cls}"
    PathUtil.auto_create_folder_path(
        frame_cfg.temp_folder_path, frame_cfg.archive_folder_path
    )
    log_filepath = f"{frame_cfg.temp_folder_path}/{frame_cfg.ID}.log"
    if frame_cfg['LOG_WITHOUT_DATE']:
        cfg.logger = Logger(filepath=log_filepath, fmt='[%(levelname)s]: %(message)s', date_fmt=None).get_std_logger()
    else:
        cfg.logger = Logger(filepath=log_filepath).get_std_logger()

    if frame_cfg['DISABLE_TQDM_BAR'] is True:
        from tqdm import tqdm
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
