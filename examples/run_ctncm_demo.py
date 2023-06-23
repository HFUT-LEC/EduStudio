import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist_0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'KTInterTrainFmt',
        'batch_size': 32,
        'eval_batch_size': 32,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterCptUnfoldDataTPL',
        'M2C_BuildSeqInterFeats': {
            'seed': 2023,
            'divide_by': 'stu',
            'window_size': 100,
            "divide_scale_list": [7,1,2],
            "extra_inter_feats": ['start_timestamp:float', 'cpt_unfold:token']
        }
    },
    model_cfg_dict={
        'cls': 'CT_NCM',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'],
    }
)
