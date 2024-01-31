import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='ASSIST_1213',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'early_stop_metrics': [('rmse','min')],
        'best_epoch_metric': 'rmse',
        'batch_size': 512
    },
    datatpl_cfg_dict={
        'cls': 'MGCDDataTPL',
        # 'load_data_from': 'rawdata',
        # 'raw2mid_op': 'R2M_ASSIST_1213'
    },
    modeltpl_cfg_dict={
        'cls': 'MGCD',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL'],
        'BinaryClassificationEvalTPL': {
            'use_metrics': ['rmse']
        }
    }
)
