import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-2015-cpt',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 32,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmt',
        'window_size': 200,
        'is_dataset_divided': True
    },
    model_cfg_dict={
        'cls': 'KQN',
        'emb_size': 128,
        'rnn_hidden_size': 128,
        'mlp_hidden_size': 128
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
