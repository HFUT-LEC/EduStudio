from .base_evaltpl import BaseEvalTPL
import torch
import pandas as pd
import numpy as np
from edustudio.utils.common import tensor2npy
from edustudio.utils.callback import ModeState
from tqdm import tqdm


class InterpretabilityEvalTPL(BaseEvalTPL):
    """Student Cogntive Representation Interpretability Evaluation
    """
    default_cfg = {
        'use_metrics': ['doa_all'],
        'test_only_metrics': ['doa_all']
    }
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = self.__class__.__name__

    def eval(self, stu_stats:np.ndarray, Q_mat:np.ndarray, **kwargs):
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_metrics', {})
        for metric_name in self.evaltpl_cfg[self.name]['use_metrics']:
            if metric_name not in ignore_metrics:
                if metric_name in self.evaltpl_cfg[self.name]['test_only_metrics'] and \
                    self.callback_list.mode_state != ModeState.END:
                   continue
               
                if metric_name == "doa_all":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        self.stu_id_total,
                        self.exer_id_total,
                        self.label_id_total,
                        user_emb=stu_stats,
                        Q_mat=Q_mat
                    )
                elif metric_name == "doa_train_val":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        self.stu_id_train_val,
                        self.exer_id_train_val,
                        self.label_id_train_val,
                        Q_mat=Q_mat
                    )
                elif metric_name == "doa_test":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        self.test_loader.dataset.dict_main['stu_id'],
                        self.test_loader.dataset.dict_main['exer_id'],
                        self.test_loader.dataset.dict_main['label'],
                        user_emb=stu_stats,
                        Q_mat=Q_mat
                    )
                elif metric_name == "doc_all":
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
        if metric == "doa_all":
            return self.doa_report
        elif metric == "doa_test":
            return self.doa_report
        elif metric == "doa_train_val":
            return self.doa_report
        elif metric == "doc_all":
            return self.degree_of_consistency
        elif metric == "doc_test":
            return self.degree_of_consistency
        elif metric == "doc_train_val":
            return self.degree_of_consistency
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

    def doa_report(self, stu_id, exer_id, label, user_emb, Q_mat):
        knowledges = []
        knowledge_item = []
        knowledge_user = []
        knowledge_truth = []
        knowledge_theta = []
        for user, item, score in zip(stu_id, exer_id, label):
            theta = user_emb[user]
            knowledge = Q_mat[item]
            if isinstance(theta, list) or isinstance(theta, np.ndarray):
                for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                    if knowledge_i == 1: 
                        knowledges.append(i) # 知识点ID
                        knowledge_item.append(item) # Item ID
                        knowledge_user.append(user) # User ID
                        knowledge_truth.append(score) # score
                        knowledge_theta.append(theta_i) # matser
            else:  # pragma: no cover
                for i, knowledge_i in enumerate(knowledge):
                    if knowledge_i == 1:
                        knowledges.append(i)
                        knowledge_item.append(item)
                        knowledge_user.append(user)
                        knowledge_truth.append(score)
                        knowledge_theta.append(theta)

        knowledge_df = pd.DataFrame({
            "knowledge": knowledges,
            "user_id": knowledge_user,
            "item_id": knowledge_item,
            "score": knowledge_truth,
            "theta": knowledge_theta
        })
        knowledge_ground_truth = []
        knowledge_prediction = []
        for _, group_df in knowledge_df.groupby("knowledge"):
            _knowledge_ground_truth = []
            _knowledge_prediction = []
            for _, item_group_df in group_df.groupby("item_id"):
                _knowledge_ground_truth.append(item_group_df["score"].values)
                _knowledge_prediction.append(item_group_df["theta"].values)
            knowledge_ground_truth.append(_knowledge_ground_truth)
            knowledge_prediction.append(_knowledge_prediction)

        return self.doa_eval(knowledge_ground_truth, knowledge_prediction)


    def doa_eval(self, y_true, y_pred):
        doa = []
        doa_support = 0
        z_support = 0
        for knowledge_label, knowledge_pred in zip(y_true, y_pred):
            _doa = 0
            _z = 0
            for label, pred in zip(knowledge_label, knowledge_pred):
                if sum(label) == len(label) or sum(label) == 0:
                    continue
                pos_idx = []
                neg_idx = []
                for i, _label in enumerate(label): # 找出所有(1, 0) pair
                    if _label == 1:
                        pos_idx.append(i)
                    else:
                        neg_idx.append(i)
                pos_pred = pred[pos_idx]
                neg_pred = pred[neg_idx]
                invalid = 0
                for _pos_pred in pos_pred:
                    _doa += len(neg_pred[neg_pred < _pos_pred])
                    invalid += len(neg_pred[neg_pred == _pos_pred])
                _z += (len(pos_pred) * len(neg_pred)) - invalid
            if _z > 0:
                doa.append(_doa / _z)
                z_support += _z # 有效pair个数
                doa_support += 1 # 有效doa
        return float(np.mean(doa))


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

    """
    reference code: https://github.com/CSLiJT/ID-CDF/blob/main/tools.py
    """
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