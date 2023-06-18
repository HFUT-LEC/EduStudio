import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio
from edustudio.datafmt.CD import MgcdDataFmt

run_edustudio(
    dataset='assist-2012',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'CDInterTrainFmt',
        'device': 'cpu',
        'best_epoch_metric': 'rmse',
        'early_stop_metrics': [('rmse','min')],
    },
    datafmt_cfg_dict={
        'cls': MgcdDataFmt,
        'is_dataset_divided': False,
        'divide_scale_list': (6, 2, 2),
    },
    model_cfg_dict={
        'cls': 'MGCD',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
