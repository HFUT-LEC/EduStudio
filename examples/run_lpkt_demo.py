import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist_0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 32,
    },
    datafmt_cfg_dict={
        'cls': 'LPKTDataTPL',
        'load_data_from': 'rawdata',
        'raw2mid_op': 'R2M_ASSIST_0910',
        'is_save_cache': False,
    },
    model_cfg_dict={
        'cls': 'LPKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
