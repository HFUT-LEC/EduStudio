import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio
from edustudio.datafmt.CD import CDInterDataFmtExtendsQ

run_edustudio(
    dataset='assist-0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'CDInterTrainFmt',
    },
    datafmt_cfg_dict={
        'cls': CDInterDataFmtExtendsQ
    },
    model_cfg_dict={
        'cls': 'NCDM',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt', 'CognitiveDiagnosisDEvalFmt']
    }
)
