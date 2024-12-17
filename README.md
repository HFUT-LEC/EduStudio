![logo](./assets/logo.png)
---

<p float="left">
<img src="https://img.shields.io/badge/python-v3.8+-blue">
<img src="https://img.shields.io/badge/pytorch-v1.10+-blue">
<img src="https://img.shields.io/badge/License-MIT-blue">
<img src="https://img.shields.io/github/issues/HFUT-LEC/EduStudio.svg">
<a href="https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-40372-3">
  <img src="https://img.shields.io/badge/Paper-EduStudio-blue" alt="Paper EduStudio Badge">
</a>
</p>

EduStudio is a Unified Library for Student Cognitive Modeling including Cognitive Diagnosis(CD) and Knowledge Tracing(KT) based on Pytorch.

## Navigation


| Resource Name                                                | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Eco-Repository](https://github.com/HFUT-LEC/awesome-student-cognitive-modeling) | A  repository containing resources about student cognitive modeling: [papers](https://github.com/HFUT-LEC/awesome-student-cognitive-modeling/tree/main/papers), [datasets](https://github.com/HFUT-LEC/awesome-student-cognitive-modeling/tree/main/datasets), [conferences&journals](https://github.com/HFUT-LEC/awesome-student-cognitive-modeling/tree/main/conferences%26journals) |
| [Eco-Leaderboard](https://leaderboard.edustudio.ai)          | A leaderboard demonstrating performance of implemented models |
| [EduStudio Documentation](https://edustudio.readthedocs.io/) | The document for EduStudio usage                             |
| [Reference Table](https://edustudio.readthedocs.io/en/latest/user_guide/reference_table.html) | The reference table demonstrating the corresponding templates of each model |

## Description

EduStudio first decomposes the general algorithmic workflow into six steps: `configuration reading`, `data prepration`, `model implementation`, `training control`, `model evaluation`, and `Log Storage`. Subsequently, to enhance the `reusability` and `scalability` of each step, we extract the commonalities of each algorithm at each step into individual templates for templatization.

<p align="center">
  <img src="assets/framework.png" alt="EduStudio Architecture" width="600">
  <br>
  <b>Figure</b>: Overall Architecture of EduStudio
</p>


## Quick Start

Install `EduStudio`:

```bash
pip install -U edustudio
```

Example: Run `NCDM` model:

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
        'cls': 'NCDM',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)

```

To find out which templates are used for a model, we can find in the [Reference Table](https://edustudio.readthedocs.io/en/latest/user_guide/reference_table.html)

## Citation
```
@article{Le WU:198342,
author = {Le WU, Xiangzhi CHEN, Fei LIU, Junsong XIE, Chenao XIA, Zhengtao TAN, Mi TIAN, Jinglong LI, Kun ZHANG, Defu LIAN, Richang HONG, Meng WANG},
title = {EduStudio: towards a unified library for student cognitive modeling},
publisher = {Front. Comput. Sci.},
year = {2025},
journal = {Frontiers of Computer Science},
volume = {19},
number = {8},
eid = {198342},
numpages = {0},
pages = {198342},
keywords = {open-source library;student cognitive modeling;intelligence education},
url = {https://journal.hep.com.cn/fcs/EN/abstract/article_47994.shtml},
doi = {10.1007/s11704-024-40372-3}
}
```


## License

EduStudio uses [MIT License](https://github.com/HFUT-LEC/EduStudio/blob/main/LICENSE). 
