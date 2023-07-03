# Run EduStudio

## Create a python file to run

create a python file (e.g., *run.py*) anywhere, the content is as follows:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'CDInterTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL'
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['BinaryClassificationEvalTPL', 'CognitiveDiagnosisEvalTPL'],
    }
)
```

Then run the following command:

```bash
python run.py
```

## Run with command

You can run the following command with parameters based on created file above.

```bash
cd examples
python run.py -dt ASSIST_0910 --modeltpl_cfg.cls NCDM --traintpl_cfg.batch_size 512
```

## Run with config file

create a yaml file `conf/ASSIST_0910/NCDM.yaml`:
```yaml
datatpl_cfg:
  cls: CDInterDataTPL

traintpl_cfg:
  cls: CDTrainTPL
  batch_size: 512

modeltpl_cfg:
  cls: NCDM

evaltpl_cfg:
  clses: [BinaryClassificationEvalTPL, CognitiveDiagnosisEvalTPL]
```

then, run command:

```bash
cd examples
python run.py -dt ASSIST_0910 -f NCDM.yaml
```
