from .base_evaltpl import BaseEvalTPL
import torch
import pandas as pd
import numpy as np
from edustudio.utils.common import tensor2npy
from edustudio.utils.callback import ModeState
from tqdm import tqdm


"""
reference code: https://github.com/CSLiJT/ID-CDF/blob/main/tools.py
"""

class IdentifiabilityEvalTPL(BaseEvalTPL):
    default_cfg = {
        'use_metrics': ['doc_all'], # Degree of Consistency
        'test_only_metrics': ['doc_all']
    }


    def eval(self, stu_stats:np.ndarray, Q_mat:np.ndarray, **kwargs):
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_metrics', {})
        for metric_name in self.evaltpl_cfg[self.name]['use_metrics']:
            if metric_name not in ignore_metrics:
                if metric_name in self.evaltpl_cfg[self.name]['test_only_metrics'] and \
                    self.callback_list.mode_state != ModeState.END:
                   continue
               
                """
                theta_mat: np.array, user_know_hit: np.array, 
                log_mat: np.array, Q_mat: np.array, know_list = None):
                """
                if metric_name == "doc_all":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        stu_stats,
                        self._gen_hit_mat(
                            self.stu_id_total,
                            self.exer_id_total,
                            Q_mat                        
                        ),
                        self.log_mat_total,
                        Q_mat
                    )
                elif metric_name == "doc_train_val":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        stu_stats,
                        self._gen_hit_mat(
                            self.stu_id_train_val,
                            self.exer_id_train_val,
                            Q_mat 
                        ),
                        self.log_mat_train_val,
                        Q_mat
                    )
                elif metric_name == "doc_test":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        stu_stats,
                        self._gen_hit_mat(
                            self.test_loader.dataset.dict_main['stu_id'],
                            self.test_loader.dataset.dict_main['exer_id'],
                            Q_mat
                        ),
                        self.log_mat_test,
                        Q_mat
                    )
                else:
                    raise ValueError(f"unknown metric_name: {metric_name}")
        return metric_result

    def _get_metrics(self, metric):
        if metric == "doc_all":
            return self.degree_of_consistency
        elif metric == "doc_test":
            return self.degree_of_consistency
        elif metric == "doc_train_val":
            return self.degree_of_consistency
        else:
            raise NotImplementedError
    
    def degree_of_consistency(self, theta_mat: np.array, user_know_hit: np.array, \
        log_mat: np.array, Q_mat: np.array, know_list = None):
        '''
        theta_mat: (n_user, n_know): the diagnostic result matrix
        user_know_hit: (n_user, n_know): the (i,j) element indicate \
            the number of hits of the i-th user on the j-th attribute
        log_mat: (n_user, n_exer): the matrix indicating whether the \
            student has correctly answered the exercise (+1) or not(-1) 
        Q_mat: (n_exer, n_know)
        '''
        n_user, n_know = theta_mat.shape 
        n_exer = log_mat.shape[1]
        doa_all = []
        know_list = list(range(n_know)) if know_list is None else know_list
        for know_id in tqdm(know_list, desc='compute_DOC', ncols=100):
            Z = 1e-9
            dm = 0
            exer_list = np.where(Q_mat[:,know_id] > 0)[0]
            user_list = np.where(user_know_hit[:,know_id]>0)[0]
            n_u_k = len(user_list)
            # pbar = tqdm(total = n_u_k * (n_u_k - 1), desc='know_id = %d'%know_id)
            for a in user_list:
                for b in user_list:
                    # if m_ak != m_bk, then either m_ak > m_bk or m_bk > m_ak
                    if a == b:
                        continue
                    Z += (theta_mat[a, know_id] > theta_mat[b, know_id])
                    nab = 1e-9
                    dab = 1e-9
                    for exer_id in exer_list:
                        Jab = (log_mat[a,exer_id] * log_mat[b,exer_id] != 0)
                        nab += Jab * (log_mat[a, exer_id] > log_mat[b, exer_id])
                        dab += Jab * (log_mat[a, exer_id] != log_mat[b, exer_id])
                    dm += (theta_mat[a, know_id] > theta_mat[b, know_id]) * nab / dab 
                    # pbar.update(1)

            doa = dm / Z 
            doa_all.append(doa)
        return np.mean(doa_all)
        
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

            self.stu_id_train_val = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['stu_id'],
                self.valid_loader.dataset.dict_main['stu_id'],
            ]))
            self.exer_id_train_val = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['exer_id'],
                self.valid_loader.dataset.dict_main['exer_id'],
            ]))

            self.label_id_train_val = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_main['label'],
                self.valid_loader.dataset.dict_main['label'],
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

            self.stu_id_train_val = tensor2npy(self.train_loader.dataset.dict_main['stu_id'])
            self.exer_id_train_val = tensor2npy(self.train_loader.dataset.dict_main['exer_id'])
            self.label_id_train_val = tensor2npy(self.train_loader.dataset.dict_main['label'])

        self.log_mat_total = self._gen_log_mat(self.stu_id_total, self.exer_id_total, self.label_id_total)
        self.log_mat_train_val = self._gen_log_mat(self.stu_id_train_val, self.exer_id_train_val, self.label_id_train_val)
        self.log_mat_test = self._gen_log_mat(
            tensor2npy(self.test_loader.dataset.dict_main['stu_id']),
            tensor2npy(self.test_loader.dataset.dict_main['exer_id']),
            tensor2npy(self.test_loader.dataset.dict_main['label']),
        )

    def _gen_log_mat(self, uid, iid, label):
        n_stu = self.datatpl_cfg['dt_info']['stu_count']
        n_exer = self.datatpl_cfg['dt_info']['exer_count']
        label = label.copy()
        label[label == 0] = -1
        log_mat = np.zeros((n_stu, n_exer), dtype=np.float32)
        log_mat[uid, iid] = label
        return log_mat


    def _gen_hit_mat(self, uid, iid, Q_mat):
        n_stu = self.datatpl_cfg['dt_info']['stu_count']
        n_exer = self.datatpl_cfg['dt_info']['exer_count']
        n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

        tmp_df = pd.DataFrame({
            'uid': uid,
            'iid': iid
        })

        # assert tmp_df['uid'].nunique() == n_stu

        hit_df = tmp_df.groupby('uid').agg(list).apply(lambda x: list(Q_mat[np.array(x['iid'])].sum(axis=0)), axis=1)
        if tmp_df['uid'].nunique() != n_stu:
            zeros = list(np.zeros((n_cpt, )))
            new_index = list(range(n_stu))
            hit_df = hit_df.reindex(new_index, fill_value=zeros)
        hit_df = hit_df.sort_index()

        return np.array(hit_df.to_list())
