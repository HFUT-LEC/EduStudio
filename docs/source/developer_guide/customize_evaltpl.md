Customize Evaluation Template
======================
Here, we present how to develop a new Evaluation Template, and apply it into EduStudio.
EduStudio provides the EvalTPL Protocol in ``EduStudio.edustudio.evaltpl.baseevaltpl.BaseEvalTPL`` (``BaseEvalTPL``).

EvalTPL Protocol
----------------------

### BaseEvalTPL
The protocols in ``BaseEvalTPL`` are listed as follows.

| name              | description               | type               | note                    |
| ----------------- | ------------------------- | ------------------ | ----------------------- |
| default_cfg       | the default configuration | class variable     |                         |
| eval              | calculate metric results  | function interface | implemented by subclass |
| _check_params     | check parameters          | function interface | implemented by subclass |
| set_callback_list | set callback list         | function interface | implemented by subclass |
| set_dataloaders   | set dataloaders           | function interface | implemented by subclass |
| add_extra_data    | add extra data            | function interface | implemented by subclass |



EvalTPLs
----------------------

EduStudio provides ``BinaryClassificationEvalTPL`` and ``CognitiveDiagnosisEvalTPL``, which inherent ``BaseEvalTPL``.

### BinaryClassificationEvalTPL
This EvalTPL is for the model evaluation using binary classification metrics.
The protocols in ``BinaryClassificationEvalTPL`` are listed as follows.


### CognitiveDiagnosisEvalTPL
This EvalTPL is for the model evaluation for interpretability. It uses states of students and Q matrix for ``eval``, which are domain-specific in student assessment.

## Develop a New EvalTPL in EduStudio

If you want to develop a new EvalTPl in EduStudio, you can inherent ``BaseEvalTPL`` and revise ``eval`` method.

### Example

```python
from .base_evaltpl import BaseEvalTPL
from sklearn.metrics import accuracy_score, coverage_error

class NewEvalFmt(BaseEvalFmt):
    default_cfg = {
        'use_metrics': ['acc', 'coverage_error']
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def eval(self, y_pd, y_gt, **kwargs):
        if not isinstance(y_pd, np.ndarray): y_pd = tensor2npy(y_pd)
        if not isinstance(y_gt, np.ndarray): y_gt = tensor2npy(y_gt)
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_metrics', {})
        for metric_name in self.evalfmt_cfg[self.__class__.__name__]['use_metrics']:
            if metric_name not in ignore_metrics:
                metric_result[metric_name] = self._get_metrics(metric_name)(y_gt, y_pd)
        return metric_result

    def _get_metrics(self, metric):
        if metric == "acc":
            return lambda y_gt, y_pd: accuracy_score(y_gt, np.where(y_pd >= 0.5, 1, 0))
        elif metric == 'coverage_error':
            return lambda y_gt, y_pd: coverage_error(y_gt, y_pd)
        else:
            raise NotImplementedError
```
