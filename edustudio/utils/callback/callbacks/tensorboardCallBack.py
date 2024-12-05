# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

from .callback import Callback

class TensorboardCallback(Callback):
    def __init__(self, log_dir, comment=''):
        super(TensorboardCallback, self).__init__()
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        for name, value in logs.items():
            self.writer.add_scalar(tag=name, scalar_value=value, global_step=epoch)
