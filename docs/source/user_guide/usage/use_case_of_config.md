# Use cases about specifying configuration

## Q1: How to specify the atomic data operation config

The default_cfg of `M2C_FilterRecords4CD` is as follows:

```python
class M2C_FilterRecords4CD(BaseMid2Cache):
    default_cfg = {
        "stu_least_records": 10,
        "exer_least_records": 0,
    }

```

The following example demonstrates how to specify config of M2C_FilterRecords4CD.

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'M2C_FilterRecords4CD': {
            "stu_least_records": 20, # look here
        }
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)
```

## Q2: How to specify the config of evaluate template
The default_cfg of `PredictionEvalTPL` is as follows:
```python
class PredictionEvalTPL(BaseEvalTPL):
    default_cfg = {
        'use_metrics': ['auc', 'acc', 'rmse']
    }
```


If we want to use only auc metric, we can do:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'M2C_FilterRecords4CD': {
            "stu_least_records": 20,
        }
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
        'InterpretabilityEvalTPL': {
            'use_metrics': {"auc"} # look here
        }
    }
)
```
