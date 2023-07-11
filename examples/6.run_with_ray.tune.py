# run following after installed edustudio

from edustudio.quickstart import run_edustudio
from ray import tune
import ray
ray.init(num_cpus=4, num_gpus=1)


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
    print(g_args)
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
    return res


search_space= {
    'traintpl_cfg.cls': tune.grid_search(['EduTrainTPL']),
    'datatpl_cfg.cls': tune.grid_search(['CDInterExtendsQDataTPL']),
    'modeltpl_cfg.cls': tune.grid_search(['KaNCD']),
    'evaltpl_cfg.clses': tune.grid_search([['BinaryClassificationEvalTPL', 'CognitiveDiagnosisEvalTPL']]),
    

    'traintpl_cfg.batch_size': tune.grid_search([256,]),
    'traintpl_cfg.epoch_num': tune.grid_search([2]),
    'traintpl_cfg.device': tune.grid_search(["cuda:0"]),
    'modeltpl_cfg.emb_dim': tune.grid_search([20,40]),
    'frame_cfg.DISABLE_LOG_STDOUT': tune.grid_search([False]),
}

tuner = tune.Tuner(
    tune.with_resources(objective_function, {"gpu": 1}), param_space=search_space, tune_config=tune.TuneConfig(max_concurrent_trials=1),
) 
results = tuner.fit()

print("=="*10)
print(results.get_best_result(metric="auc", mode="max").config)
