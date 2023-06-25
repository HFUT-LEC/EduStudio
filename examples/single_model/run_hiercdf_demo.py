import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='JunyiExerAsCpt',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'CDInterTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'HierCDFDataTPL'
    },
    modeltpl_cfg_dict={
        'cls': 'HierCDF',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL'],
    }
)