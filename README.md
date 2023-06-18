![logo](./assets/logo.png)
---

<p float="left">
<img src="https://img.shields.io/badge/python-v3.8+-blue">
<img src="https://img.shields.io/badge/pytorch-v1.10+-blue">
<img src="https://img.shields.io/badge/License-MIT-blue">
<img src="https://img.shields.io/github/issues/HFUT-LEC/EduStudio.svg">
</p>

EduStudio is a Unified and Templatized Framework for Student Assessment Models including Cognitive Diagnosis(CD) and Knowledge Tracing(KT) based on Pytorch.

# Description

![Overall Framework](./assets/framework.svg)

## Quick Start

Install `EduStudio`:

```bash
pip install edustudio
```

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

## License

EduStudio uses [MIT License](https://github.com/HFUT-LEC/EduStudio/blob/main/LICENSE). 

