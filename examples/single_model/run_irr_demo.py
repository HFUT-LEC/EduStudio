import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'lr': 0.0001,
    },
    datatpl_cfg_dict={
        'cls': 'IRRDataTPL',
    },
    modeltpl_cfg_dict={
        'cls': 'IRR',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL'],
    }
)
