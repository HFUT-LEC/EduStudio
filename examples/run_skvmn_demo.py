import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-2015-cpt',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 64,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmt',
        'window_size': 100,
        'is_dataset_divided': True
    },
    model_cfg_dict={
        'cls': 'SKVMN',
        'memory_size': 50,
        'embed_dim': 200,
        'param_init_type': 'kaiming_normal',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
