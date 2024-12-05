# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import os
import sys
import time
from functools import wraps
import pytz
import datetime
import json
import random
import torch
import numpy as np
import logging


tensor2npy = lambda x: x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy()
tensor2cpu = lambda x: x.cpu() if x.is_cuda else x


def set_same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PathUtil(object):
    @staticmethod
    def auto_create_folder_path(*args):
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path)

    @staticmethod
    def get_main_folder_path():
        try:
            __IPYTHON__
        except NameError:
            return os.path.realpath(os.path.dirname(os.path.abspath(sys.argv[0])))
        return os.getcwd()

    @staticmethod
    def check_path_exist(*args):
        for path in args:
            if not os.path.exists(path):
                raise FileNotFoundError(os.path.realpath(path) + " not exists")


class IOUtil(object):
    @staticmethod
    def read_json_file(filepath, encoding='utf-8'):
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def write_json_file(filepath, data, encoding='utf-8', ensure_ascii=False, indent=2):
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


class IDUtil(object):
    @staticmethod
    def get_random_id_bytime():
        tz = pytz.timezone('Asia/Shanghai')
        return datetime.datetime.now(tz).strftime("%Y-%m-%d-%H%M%S")


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.
    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


class DecoratorTimer:
    def __init__(self):
        self.logger = logging.getLogger("edustudio")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.info(f'Function:{func.__name__} start running...')
            start_time = time.time()
            temp = func(*args, **kwargs)
            end_time = time.time()
            self.logger.info(f'Function:{func.__name__} running time: {end_time - start_time:.4f} sec')
            return temp

        return wrapper