from .base_evaltpl import BaseEvalTPL
import torch
import pandas as pd
import numpy as np
from edustudio.utils.common import tensor2npy
from edustudio.utils.callback import ModeState
from tqdm import tqdm


class IdentifiabilityEvalTPL(BaseEvalTPL):
    default_cfg = {
        'use_metrics': ['IDS'], # Identifiability Score
        'test_only_metrics': ['IDS']
    }

    def eval(self, stu_stats:np.ndarray,  **kwargs):
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_metrics', {})
        for metric_name in self.evaltpl_cfg[self.name]['use_metrics']:
            if metric_name not in ignore_metrics:
                if metric_name in self.evaltpl_cfg[self.name]['test_only_metrics'] and \
                    self.callback_list.mode_state != ModeState.END:
                   continue
                if metric_name == "IDS":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                       self.log_mat_total, stu_stats
                    )
                else:
                    raise ValueError(f"unknown metric_name: {metric_name}")
        return metric_result

    def _get_metrics(self, metric):
        if metric == "IDS":
            return self.get_IDS
        else:
            raise NotImplementedError
    
    def set_dataloaders(self, train_loader, test_loader, valid_loader=None):
        super().set_dataloaders(train_loader, test_loader, valid_loader)

        if self.valid_loader:
            
            self.stu_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['stu_id'],
                self.valid_loader.dataset.dict_main['stu_id'],
                self.test_loader.dataset.dict_main['stu_id'],
            ]))
            self.exer_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['exer_id'],
                self.valid_loader.dataset.dict_main['exer_id'],
                self.test_loader.dataset.dict_main['exer_id'],
            ]))

            self.label_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['label'],
                self.valid_loader.dataset.dict_main['label'],
                self.test_loader.dataset.dict_main['label'],
            ]))

        else:
            self.stu_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['stu_id'],
                self.test_loader.dataset.dict_main['stu_id'],
            ]))
            self.exer_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['exer_id'],
                self.test_loader.dataset.dict_main['exer_id'],
            ]))

            self.label_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['label'],
                self.test_loader.dataset.dict_main['label'],
            ]))

        self.log_mat_total = self._gen_log_mat(self.stu_id_total, self.exer_id_total, self.label_id_total)
  
    def _gen_log_mat(self, uid, iid, label):
        n_stu = self.datatpl_cfg['dt_info']['stu_count']
        n_exer = self.datatpl_cfg['dt_info']['exer_count']
        label = label.copy()
        label[label == 0] = -1
        log_mat = np.zeros((n_stu, n_exer), dtype=np.float32)
        log_mat[uid, iid] = label
        return log_mat

    def get_IDS(self, R, T):
        """
        R: interaction matrix
        T: student proficiency matrix
        """
        count = 0
        val = 0.0
        n_stu = R.shape[0]
        assert R.shape[0] == T.shape[0]
        for s_i in range(n_stu):
            for s_j in range(s_i+1, n_stu):
                if not np.all(R[s_i] == R[s_j]):
                    continue
                count += 1
                val += (1+np.abs(T[s_i] - T[s_j]).sum())**2
        return (count / val).item()
