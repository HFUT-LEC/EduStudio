import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from edustudio.quickstart import run_edustudio
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


def deliver_cfg(args):
    g_args = {
        'traintpl_cfg': {},
        'datatpl_cfg': {},
        'modeltpl_cfg': {},
        'evaltpl_cfg': {},
        'frame_cfg': {},
    }
    for k,v in args.items():
        g, k = k.split(".")
        assert g in g_args
        g_args[g][k] = v
    return g_args


# objective function
def objective_function(args):
    g_args = deliver_cfg(args)
    cfg, res = run_edustudio(
        dataset='FrcSub',
        cfg_file_name=None,
        traintpl_cfg_dict=g_args['traintpl_cfg'],
        datatpl_cfg_dict=g_args['datatpl_cfg'],
        modeltpl_cfg_dict=g_args['modeltpl_cfg'],
        evaltpl_cfg_dict=g_args['evaltpl_cfg'],
        frame_cfg_dict=g_args['frame_cfg'],
        return_cfg_and_result=True
    )
    return res['auc']


space = {
    'traintpl_cfg.cls': hp.choice('traintpl_cfg.cls', ['GeneralTrainTPL']),
    'datatpl_cfg.cls': hp.choice('datapl_cfg.cls', ['CDInterExtendsQDataTPL']),
    'modeltpl_cfg.cls': hp.choice('modeltpl_cfg.cls', ['KaNCD']),
    'evaltpl_cfg.clses': hp.choice('evaltpl_cfg.clses', [['PredictionEvalTPL', 'InterpretabilityEvalTPL']]),
    

    'traintpl_cfg.batch_size': hp.choice('traintpl_cfg.batch_size', [256,]),
    'traintpl_cfg.epoch_num': hp.choice('traintpl_cfg.epoch_num', [2]),
    'modeltpl_cfg.emb_dim': hp.choice('modeltpl_cfg.emb_dim', [20,40])
}

best = fmin(objective_function, space, algo=tpe.suggest, max_evals=10, verbose=False)

print("=="*10)
print(best)
print(space_eval(space, best))
