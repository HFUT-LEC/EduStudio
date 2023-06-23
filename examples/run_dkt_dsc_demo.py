import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist_0910',
    cfg_file_name=None,
    datafmt_cfg_dict={
        'cls': 'DKTDSCDataTPL',
        'load_data_from': 'cachedata',
        'raw2mid_op': 'R2M_ASSIST_0910',
        'is_save_cache': True,
        'cache_id': 'dkt_dsc'
    },
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
    },
    model_cfg_dict={
        'cls': 'DKTDSC',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
