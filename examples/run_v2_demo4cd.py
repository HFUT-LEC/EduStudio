import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='ASSIST_0910',
    cfg_file_name=None,
    datafmt_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'load_data_from': 'rawdata',
        'raw2mid_op': 'R2M_ASSIST_0910',
        'is_save_cache': False,
        'mid2cache_op_seq': ["M2C_CptAsExer", 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD'],
    },
    trainfmt_cfg_dict={
        'cls': 'CDInterTrainFmt',
    },
    model_cfg_dict={
        'cls': 'IRT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
