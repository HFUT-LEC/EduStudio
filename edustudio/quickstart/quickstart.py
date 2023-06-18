from typing import Dict, Any
import importlib
import sys
import logging
import shutil
import traceback

from .parse_cfg import get_global_cfg
from .init_all import init_all

from edustudio.utils.common import UnifyConfig


def run_edustudio(
    dataset: str = None,
    cfg_file_name: str = None,  # config file
    trainfmt_cfg_dict: Dict[str, Any] = {},  # parameters dictionary
    datafmt_cfg_dict:  Dict[str, Any] = {},  # parameters dictionary
    evalfmt_cfg_dict:  Dict[str, Any] = {},  # parameters dictionary
    model_cfg_dict: Dict[str, Any] = {},  # parameters dictionary
    frame_cfg_dict:  Dict[str, Any] = {},  # parameters dictionary
    return_cfg_and_result: bool = False,
):
    cfg: UnifyConfig = get_global_cfg(
        dataset, cfg_file_name, trainfmt_cfg_dict,
        datafmt_cfg_dict, evalfmt_cfg_dict, model_cfg_dict, frame_cfg_dict
    )
    init_all(cfg)
    try:
        cfg.logger.info("====" * 15)
        cfg.logger.info(f"[ID]: {cfg.frame_cfg.ID}")
        cfg.logger.info(f"[DATASET]: {cfg.dataset}")
        cfg.logger.info(f"[ARGV]: {sys.argv}")
        cfg.logger.info(f"[ALL_CFG]: \n{cfg.dump_fmt()}")
        cfg.dump_file(f"{cfg.frame_cfg.temp_folder_path}/cfg.json")
        cfg.logger.info("====" * 15)
        if isinstance(cfg.trainfmt_cfg['cls'], str):
            cls = importlib.import_module('edustudio.trainfmt').\
                __getattribute__(cfg.trainfmt_cfg['cls'])
        else:
            cls = cfg.trainfmt_cfg['cls']
        trainfmt = cls(cfg)
        trainfmt.start()
        cfg.logger.info(f"Task: {cfg.frame_cfg.ID} Completed!")
        logging.shutdown()
        shutil.move(cfg.frame_cfg.temp_folder_path, cfg.frame_cfg.archive_folder_path)
    except Exception as e:
        cfg.logger.error(traceback.format_exc())
        raise e
    
    if return_cfg_and_result:
        return cfg, read_exp_result(cfg)


def read_exp_result(cfg):
    with open(f"{cfg.archive_folder_path}/result.json", 'r', encoding='utf-8') as f:
        import json
        return json.load(f)


if __name__ == "__main__":
    run_edustudio()
