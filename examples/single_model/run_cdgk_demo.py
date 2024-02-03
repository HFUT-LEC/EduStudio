import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'batch_size': 1024
    },
    datatpl_cfg_dict={
        'cls': 'CDGKDataTPL',
        'M2C_CDGK_OP': {
            'subgraph_count': 1, 
        }
    },
    modeltpl_cfg_dict={
        'cls': 'CDGK_MULTI',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL'],
    }
)
