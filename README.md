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


<p align="center">
  <img src="assets/framework.svg" alt="EduStudio Architecture" width="600">
  <br>
  <b>Figure</b>: EduStudio Overall Architecture
</p>

## Quick Start

Install `EduStudio`:

```bash
pip install edustudio
```

Example: Run `KaNCD` model:

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='ASSIST_0910',
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

## License

EduStudio uses [MIT License](https://github.com/HFUT-LEC/EduStudio/blob/main/LICENSE). 

