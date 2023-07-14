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
    traintpl_cfg_dict: Dict[str, Any] = {},  # parameters dictionary
    datatpl_cfg_dict:  Dict[str, Any] = {},  # parameters dictionary
    evaltpl_cfg_dict:  Dict[str, Any] = {},  # parameters dictionary
    modeltpl_cfg_dict: Dict[str, Any] = {},  # parameters dictionary
    frame_cfg_dict:  Dict[str, Any] = {},  # parameters dictionary
    return_cfg_and_result: bool = False,
):
    """The quick start API to run edustudio

    Args:
        dataset (str, optional):dataset name. Defaults to None.
        cfg_file_name (str, optional): config file name. Defaults to None.
        traintpl_cfg_dict (Dict[str, Any]): parameter dict of training template
        datatpl_cfg_dict (Dict[str, Any]): parameter dict of data template
        evaltpl_cfg_dict (Dict[str, Any]): parameter dict of evaluate template
        modeltpl_cfg_dict (Dict[str, Any]): parameter dict of model template
        frame_cfg_dict (Dict[str, Any]): parameter dict of framework template

    Returns:
        tuple: the global config object and experimental result
    """
    cfg: UnifyConfig = get_global_cfg(
        dataset, cfg_file_name, traintpl_cfg_dict,
        datatpl_cfg_dict, evaltpl_cfg_dict, modeltpl_cfg_dict, frame_cfg_dict
    )
    init_all(cfg)
    ret = None
    try:
        cfg.logger.info("====" * 15)
        cfg.logger.info(f"[ID]: {cfg.frame_cfg.ID}")
        cfg.logger.info(f"[DATASET]: {cfg.dataset}")
        cfg.logger.info(f"[ARGV]: {sys.argv}")
        cfg.logger.info(f"[ALL_CFG]: \n{cfg.dump_fmt()}")
        cfg.dump_file(f"{cfg.frame_cfg.temp_folder_path}/cfg.json")
        cfg.logger.info("====" * 15)
        if isinstance(cfg.traintpl_cfg['cls'], str):
            cls = importlib.import_module('edustudio.traintpl').\
                __getattribute__(cfg.traintpl_cfg['cls'])
        else:
            cls = cfg.traintpl_cfg['cls']
        traintpl = cls(cfg)
        traintpl.start()
        cfg.logger.info(f"Task: {cfg.frame_cfg.ID} Completed!")
        
        for handler in cfg.logger.handlers:
            handler.close()
            cfg.logger.removeHandler(handler)

        for handler in cfg.logger.handlers:
            handler.close()
            cfg.logger.removeHandler(handler)

        shutil.move(cfg.frame_cfg.temp_folder_path, cfg.frame_cfg.archive_folder_path)
    except Exception as e:
        cfg.logger.error(traceback.format_exc())
        raise e
    
    if return_cfg_and_result: ret = (cfg, read_exp_result(cfg))
    return ret

def read_exp_result(cfg):
    with open(f"{cfg.frame_cfg.archive_folder_path}/{cfg.frame_cfg.ID}/result.json", 'r', encoding='utf-8') as f:
        import json
        return json.load(f)


if __name__ == "__main__":
    run_edustudio()
