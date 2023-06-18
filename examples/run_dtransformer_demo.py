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
        'batch_size': 8,
        'eval_batch_size': 8,
        'weight_decay': 1e-5,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmtCptUnfold',
        'window_size': 100,
        'is_dataset_divided': True
    },
    model_cfg_dict={
        'cls': 'DTransformer',
        'n_knowledges': 32,
        'projection_alhead_cl': 1,
        'cl_loss': True,
        'hard_negative': True,
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
