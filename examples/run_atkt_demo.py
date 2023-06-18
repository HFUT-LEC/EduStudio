import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'AtktTrainFmt',
        'batch_size': 64,
        'device': 'cpu',
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmtCptUnfold',
        'window_size': 100,
        'is_dataset_divided': True,
    },
    model_cfg_dict={
        'cls': 'ATKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
