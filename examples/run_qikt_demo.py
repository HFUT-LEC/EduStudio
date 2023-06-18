import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 64,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmtExtendsQ',
        'window_size': 200,
        'is_dataset_divided': True
    },
    model_cfg_dict={
        'cls': 'QIKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
