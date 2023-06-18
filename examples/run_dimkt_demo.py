import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-2012',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 64,
        'device': 'cuda:1',#cuda:1
    },
    datafmt_cfg_dict={
        'cls': 'DIMKTDataFmt',
        'window_size': 200,
        'is_dataset_divided': False
    },
    model_cfg_dict={
        'cls': 'DIMKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
