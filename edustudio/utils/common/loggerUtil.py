# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import logging
import pytz
import sys
import os


class Logger(object):
    def __init__(self, filepath: str = None,
                 fmt: str = "%(asctime)s[%(levelname)s]: %(message)s",
                 date_fmt: str = "%Y-%m-%d %H:%M:%S",
                 timezone: str = "Asia/Shanghai",
                 level=logging.DEBUG,
                 DISABLE_LOG_STDOUT=False
                 ):
        self.timezone = timezone
        if filepath:
            dir_name = os.path.dirname(filepath)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        self.logger = logging.getLogger("edustudio")
        self.logger.setLevel(level)
        formatter = logging.Formatter(
            fmt, datefmt=date_fmt
        )
        formatter.converter = self.converter

        if filepath:
            # write into file
            fh = logging.FileHandler(filepath, mode='a+', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        flag = False
        for handler in self.logger.handlers:
            if type(handler) is logging.StreamHandler:
                flag = True
        if flag is False and DISABLE_LOG_STDOUT is False:
            self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()

    def converter(self, sec):
        tz = pytz.timezone(self.timezone)
        dt = pytz.datetime.datetime.fromtimestamp(sec, tz)
        return dt.timetuple()

    def get_std_logger(self):
        return self.logger
