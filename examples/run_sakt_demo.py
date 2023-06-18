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
        'batch_size': 256,  
        'epoch_num':200,
    },
    datafmt_cfg_dict={
        'cls': 'KTInterDataFmt',
        'window_size': 100,  
        'is_dataset_divided': True,
        'divide_scale_list': (8, 1, 1),
    },
    model_cfg_dict={
        'cls': 'SAKT',
        'emb_size': 128,  
        'param_init_type': 'kaiming_normal'
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt'], 
    }
)