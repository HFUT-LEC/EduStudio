# Quick Start

Here is a quick-start example for using EduStudio.

## Create a python file to run

create a python file (e.g., *run.py*) anywhere, the content is as follows:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL'
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)
```

Then run the following command:

```bash
python run.py
```

To find out which templates are used for a model, we can find in the [Reference Table](https://edustudio.readthedocs.io/en/latest/user_guide/reference_table.html)
