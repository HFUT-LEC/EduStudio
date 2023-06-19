# Quick Start

Example: Run `NCDM` model:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='assist-0910',
    cfg_file_name=None,
    trainfmt_cfg_dict={
        'cls': 'CDInterTrainFmt',
    },
    datafmt_cfg_dict={
        'cls': 'CDInterDataFmtExtendsQ',
    },
    model_cfg_dict={
        'cls': 'NCDM',
    },
    evalfmt_cfg_dict={
        'clses': ['BinaryClassificationEvalFmt']
    }
)
```
