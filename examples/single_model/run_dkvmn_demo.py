import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='ASSIST_0910',
    cfg_file_name=None,
    datatpl_cfg_dict={
        'cls': 'KTInterExtendsQDataTPL',
    },
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'eval_batch_size': 256,
    },
    modeltpl_cfg_dict={
        'cls': 'DKVMN',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL'],
    }
)


