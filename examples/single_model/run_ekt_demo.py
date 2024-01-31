import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='AAAI_2023',
    cfg_file_name=None,
    datatpl_cfg_dict={
        'cls': 'EKTDataTPL',
        'M2C_BuildSeqInterFeats': {
            'window_size': 50,
        }
    },
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'batch_size': 8,
        'eval_batch_size': 8,
    },
    modeltpl_cfg_dict={
        'cls': 'EKTM',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL'],
    }
)
