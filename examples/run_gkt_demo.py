import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='algebra-2005-cpt',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 32,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmtCptAsExer',
        'window_size': 100,
        'is_dataset_divided': True
    },
    model_cfg_dict={
        'cls': 'GKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
