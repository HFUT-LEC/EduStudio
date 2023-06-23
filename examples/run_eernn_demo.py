import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='AAAI_2023',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 4,
    },
    datafmt_cfg_dict={
        'cls': 'EERNNDataTPL',
        'load_data_from': 'cachedata',
        'raw2mid_op': 'R2M_AAAI_2023',
        'is_save_cache': True,
    },
    model_cfg_dict={
        'cls': 'EERNNA',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
