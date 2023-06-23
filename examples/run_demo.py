import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'CDInterTrainFmt',
    },
    datafmt_cfg_dict={
        'cls': "IRRDataFmt",
        'backbone_datafmt_cls': "CDInterDataFmtExtendsQ"
    },
    model_cfg_dict={
        'cls': 'IRR',
        'backbone_model_cls': 'NCDM' 
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
