# Hyper Paramter Tuning

The quick start API `run_edustudio` can be seen as the object function in various `Automatic Hyper Parameter` Toolkits. 

With the paramter `return_cfg_and_result=True`, `run_edustudio` function will return global `cfg` object and experimental result dictionary. The result is obtained by reading the `result.json` file:
```python
def read_exp_result(cfg):
    with open(f"{cfg.frame_cfg.archive_folder_path}/{cfg.frame_cfg.ID}/result.json", 'r', encoding='utf-8') as f:
        import json
        return json.load(f)
```

Here we list two demos for `Ray.Tune` and `HyperOpt`.

## Ray.Tune

```python
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
    'traintpl_cfg.cls': tune.grid_search(['GeneralTrainTPL']),
    'datatpl_cfg.cls': tune.grid_search(['CDInterExtendsQDataTPL']),
    'modeltpl_cfg.cls': tune.grid_search(['KaNCD']),
    'evaltpl_cfg.clses': tune.grid_search([['BinaryClassificationEvalTPL', 'CognitiveDiagnosisEvalTPL']]),
    

    'traintpl_cfg.batch_size': tune.grid_search([256,]),
    'traintpl_cfg.epoch_num': tune.grid_search([2]),
    'traintpl_cfg.device': tune.grid_search(["cpu"]),
    'modeltpl_cfg.emb_dim': tune.grid_search([20,40])
}

tuner = tune.Tuner(
    objective_function, param_space=search_space, tune_config=tune.TuneConfig(max_concurrent_trials=1)
) 
results = tuner.fit()

print("=="*10)
print(results.get_best_result(metric="auc", mode="max").config)
```

## HyperOpt

```python
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
    'evaltpl_cfg.clses': hp.choice('evaltpl_cfg.clses', [['BinaryClassificationEvalTPL', 'CognitiveDiagnosisEvalTPL']]),
    

    'traintpl_cfg.batch_size': hp.choice('traintpl_cfg.batch_size', [256,]),
    'traintpl_cfg.epoch_num': hp.choice('traintpl_cfg.epoch_num', [2]),
    'modeltpl_cfg.emb_dim': hp.choice('modeltpl_cfg.emb_dim', [20,40])
}

best = fmin(objective_function, space, algo=tpe.suggest, max_evals=10, verbose=False)

print("=="*10)
print(best)
print(space_eval(space, best))
```
