
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-2017',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 256,
        'device': 'cuda:1',
        'num_stop_rounds': 20,
    },
    datafmt_cfg_dict={
        # 'cls': 'KTInterDataFmt',
        'cls':'QDKTDataFmt',
        'window_size': 200,
        'is_dataset_divided': False
    },
    model_cfg_dict={
        'cls': 'QDKT',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)