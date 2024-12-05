import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'device': 'cpu'
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL'
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL', 'IdentifiabilityEvalTPL'],
        'PredictionEvalTPL': {
            'use_metrics': ['auc'],
        },
        'InterpretabilityEvalTPL': {
            'use_metrics': ['doa_all', 'doc_all'],
            'test_only_metrics': ['doa_all', 'doc_all'],
        },
        'IdentifiabilityEvalTPL': {
            'use_metrics': ['IDS']
        }
    }
)
