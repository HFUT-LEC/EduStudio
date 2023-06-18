# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from .callback import Callback
import os
import pickle


class EpochPredict(Callback):
    def __init__(self, save_folder_path, fmt="predict-{epoch}", **kwargs):
        super(EpochPredict, self).__init__()
        self.kwargs = kwargs
        self.save_folder_path = save_folder_path
        if not os.path.exists(save_folder_path):
            os.makedirs(self.save_folder_path)
        self.file_fmt = fmt + ".pkl"
        self.pd = None
        assert "{epoch}" in self.file_fmt

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        self.pd = self.model.predict(**self.kwargs)
        filepath = os.path.join(self.save_folder_path, self.file_fmt.format(epoch=epoch))
        self.to_pickle(filepath, self.pd)

    @staticmethod
    def to_pickle(filepath, obj):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
