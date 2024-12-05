from .base_evaltpl import BaseEvalTPL
import numpy as np
from edustudio.utils.common import tensor2npy
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, f1_score, label_ranking_loss, coverage_error


class PredictionEvalTPL(BaseEvalTPL):
    """Student Performance Prediction Evaluation
    """
    default_cfg = {
        'use_metrics': ['auc', 'acc', 'rmse']
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    
    def eval(self, y_pd, y_gt, **kwargs):
        if not isinstance(y_pd, np.ndarray): y_pd = tensor2npy(y_pd)
        if not isinstance(y_gt, np.ndarray): y_gt = tensor2npy(y_gt)
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_metrics', {})
        for metric_name in self.evaltpl_cfg[self.__class__.__name__]['use_metrics']:
            if metric_name not in ignore_metrics:
                metric_result[metric_name] = self._get_metrics(metric_name)(y_gt, y_pd)
        return metric_result
        
    def _get_metrics(self, metric):
        if metric == "auc":
            return roc_auc_score
        elif metric == "mse":
            return mean_squared_error
        elif metric == 'rmse':
            return lambda y_gt, y_pd: mean_squared_error(y_gt, y_pd) ** 0.5
        elif metric == "acc":
            return lambda y_gt, y_pd: accuracy_score(y_gt, np.where(y_pd >= 0.5, 1, 0))
        elif metric == "f1_macro":
            return lambda y_gt, y_pd: f1_score(y_gt, y_pd, average='macro')
        elif metric == "f1_micro":
            return lambda y_gt, y_pd: f1_score(y_gt, y_pd, average='micro')
        elif metric == "ranking_loss":
            return lambda y_gt, y_pd: label_ranking_loss(y_gt, y_pd)
        elif metric == 'coverage_error':
            return lambda y_gt, y_pd: coverage_error(y_gt, y_pd)
        elif metric == 'samples_auc':
            return lambda y_gt, y_pd: roc_auc_score(y_gt, y_pd, average='samples')
        else:
            raise NotImplementedError

