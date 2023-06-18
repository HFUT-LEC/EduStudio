import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 32,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmtCL4KT',
        'window_size': 100,
        'is_dataset_divided': True
    },
    model_cfg_dict={
        'cls': 'CL4KT',
        'emb_size': 64,
        'hidden_size': 64,

    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
