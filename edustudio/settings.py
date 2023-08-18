# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from edustudio.utils.common import PathUtil as pathUtil
from edustudio.utils.common import IDUtil as idUtil
import os

ID = idUtil.get_random_id_bytime() # RUN ID

WORK_DIR = os.getcwd()

DATA_FOLDER_PATH = f"{WORK_DIR}/data"
TEMP_FOLDER_PATH = f"{WORK_DIR}/temp"
ARCHIVE_FOLDER_PATH = f"{WORK_DIR}/archive"
CFG_FOLDER_PATH = f"{WORK_DIR}/conf"

pathUtil.auto_create_folder_path(
    TEMP_FOLDER_PATH,
    ARCHIVE_FOLDER_PATH,
    DATA_FOLDER_PATH,
    CFG_FOLDER_PATH,
)

DISABLE_TQDM_BAR = False
LOG_WITHOUT_DATE = False
TQDM_NCOLS = 100
DISABLE_LOG_STDOUT = False

curr_file_folder = os.path.dirname(__file__)
DT_INFO_FILE_PATH = os.path.realpath(curr_file_folder + "/assets/datasets.yaml")

DT_INFO_DICT = {} # additional dataset info entrypoint, example: {'ASSIST_0910': {middata_url: https://gitlab.com/hfut-lec/edudatafiles/-/raw/main/ASSIST_0910/ASSIST_0910-middata.zip} }
