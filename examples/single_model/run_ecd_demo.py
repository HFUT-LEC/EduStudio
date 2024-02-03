import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='PISA_2015_ECD',#ecd only supports PISA dataset
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'batch_size': 2048,
        'eval_batch_size': 2048,
    },
    datatpl_cfg_dict={
        'cls': 'ECDDataTPL',
    },
    modeltpl_cfg_dict={
        'cls': 'ECD_IRT',#ECD_IRT,ECD_MIRT,ECD_NCD
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL'],
    }
)
