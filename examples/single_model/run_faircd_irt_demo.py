import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='term-eng',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'AdversarialTrainTPL',
        'device': 'cpu'
    },
    datatpl_cfg_dict={
        'cls': 'FAIRCDDataTPL',
        # 'load_data_from': 'rawdata',
        # 'raw2mid_op': 'R2M_SLP_English'
    },
    modeltpl_cfg_dict={
        'cls': 'FairCD_IRT',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'FairnessEvalTPL'],
    }
)
