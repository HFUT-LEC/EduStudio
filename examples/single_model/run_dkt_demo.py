import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='ASSIST_0910',
    cfg_file_name=None,
    datafmt_cfg_dict={
        'cls': 'KTInterCptAsExerDataTPL',
    },
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
    },
    model_cfg_dict={
        'cls': 'DKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
