from .gd_traintpl import GDTrainTPL
from edustudio.utils.common import UnifyConfig, set_same_seeds, tensor2npy
from edustudio.utils.callback import ModelCheckPoint, EarlyStopping, History, BaseLogger, Callback, CallbackList
from edustudio.model import BaseModel
import torch
from typing import Sequence
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import shutil


class GeneralTrainTPL(GDTrainTPL):
    default_cfg = {
        'num_stop_rounds': 10,
        'early_stop_metrics': [('auc','max')],
        'best_epoch_metric': 'auc',
        'unsave_best_epoch_pth': True,
        'ignore_metrics_in_train': [],
        'batch_size': 32,
        'eval_batch_size': 32
    }

    def _check_params(self):
        super()._check_params()
        assert self.traintpl_cfg['best_epoch_metric'] in set(i[0] for i in self.traintpl_cfg['early_stop_metrics'])

    def one_fold_start(self, fold_id):
        super().one_fold_start(fold_id)
        # callbacks
        num_stop_rounds = self.traintpl_cfg['num_stop_rounds']
        es_metrics = self.traintpl_cfg["early_stop_metrics"]
        modelCheckPoint = ModelCheckPoint(
            es_metrics, save_folder_path=f"{self.frame_cfg.temp_folder_path}/pths/{fold_id}/"
        )
        es_cb = EarlyStopping(es_metrics, num_stop_rounds=num_stop_rounds, start_round=1)
        history_cb = History(folder_path=f"{self.frame_cfg.temp_folder_path}/history/{fold_id}", plot_curve=False)
        callbacks = [
            modelCheckPoint, es_cb, history_cb,
            BaseLogger(self.logger, group_by_contains=['loss'])
        ]
        self.callback_list = CallbackList(callbacks=callbacks, model=self.model, logger=self.logger)
        # evaltpls
        for evaltpl in self.evaltpls: 
            evaltpl.set_callback_list(self.callback_list)
            evaltpl.set_dataloaders(train_loader=self.train_loader, 
                                    valid_loader=self.valid_loader, 
                                    test_loader=self.test_loader
                                    )
        # train
        #set_same_seeds(self.traintpl_cfg['seed'])
        if self.valid_loader is not None:
            self.fit(train_loader=self.train_loader, valid_loader=self.valid_loader)
        else:
            self.logger.info("Without validation set, replace it with test set")
            self.fit(train_loader=self.train_loader, valid_loader=self.test_loader)
        
        metric_name = self.traintpl_cfg['best_epoch_metric']
        metric = [m for m in modelCheckPoint.metric_list if m.name == metric_name][0]
        if self.valid_loader is not None:
            # load best params
            fpth =  f"{self.frame_cfg.temp_folder_path}/pths/{fold_id}/best-epoch-{metric.best_epoch:03d}-for-{metric.name}.pth"
            self.model.load_state_dict(torch.load(fpth))

            metrics = self.inference(self.test_loader)
            for name in metrics: self.logger.info(f"{name}: {metrics[name]}")
            History.dump_json(metrics, f"{self.frame_cfg.temp_folder_path}/{fold_id}/result.json")
        else:
            # metrics = history_cb.log_as_time[metric.best_epoch]
            # load best params
            fpth =  f"{self.frame_cfg.temp_folder_path}/pths/{fold_id}/best-epoch-{metric.best_epoch:03d}-for-{metric.name}.pth"
            self.model.load_state_dict(torch.load(fpth))

            metrics = self.inference(self.test_loader)
            for name in metrics: self.logger.info(f"{name}: {metrics[name]}")
            History.dump_json(metrics, f"{self.frame_cfg.temp_folder_path}/{fold_id}/result.json")

        if self.traintpl_cfg['unsave_best_epoch_pth']: shutil.rmtree(f"{self.frame_cfg.temp_folder_path}/pths/")
        return metrics

    def fit(self, train_loader, valid_loader):
        self.model.train()
        optimizer = self.traintpl_cfg['optim']
        lr = self.traintpl_cfg['lr']
        weight_decay = self.traintpl_cfg['weight_decay']
        eps = self.traintpl_cfg['eps']
        self.optimizer = self._get_optim(optimizer=optimizer, lr=lr, weight_decay=weight_decay, eps=eps)
        self.callback_list.on_train_begin()
        for epoch in range(self.traintpl_cfg['epoch_num']):
            self.callback_list.on_epoch_begin(epoch + 1)
            logs = defaultdict(lambda: np.full((len(train_loader),), np.nan, dtype=np.float32))
            for batch_id, batch_dict in enumerate(
                    tqdm(train_loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[EPOCH={:03d}]".format(epoch + 1))
            ):
                batch_dict = self.batch_dict2device(batch_dict)
                loss_dict = self.model.get_loss_dict(**batch_dict)
                loss = torch.hstack([i for i in loss_dict.values() if i is not None]).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                for k in loss_dict: 
                    if loss_dict[k] is not None:
                        logs[k][batch_id] = loss_dict[k].item()

            for name in logs: logs[name] = float(np.nanmean(logs[name]))

            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                logs.update({f"{metric}": val_metrics[metric] for metric in val_metrics})

            self.callback_list.on_epoch_end(epoch + 1, logs=logs)
            if self.model.share_callback_dict.get('stop_training', False):
                break
            
        self.callback_list.on_train_end()
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        stu_id_list = list(range(len(loader)))
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            stu_id_list[idx] = batch_dict['stu_id']
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = eval_dict['y_gt'] if 'y_gt' in eval_dict else batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)
        stu_id = torch.hstack(stu_id_list)

        eval_data_dict = {
            'stu_id': stu_id,
            'y_pd': y_pd,
            'y_gt': y_gt,
        }
        if hasattr(self.model, 'get_stu_status'):
            stu_stats_list = []
            idx = torch.arange(0, self.datatpl_cfg['dt_info']['stu_count']).to(self.traintpl_cfg['device'])
            for i in range(0,self.datatpl_cfg['dt_info']['stu_count'], self.traintpl_cfg['eval_batch_size']):
                batch_stu_id = idx[i:i+self.traintpl_cfg['eval_batch_size']]
                batch = self.model.get_stu_status(batch_stu_id)
                stu_stats_list.append(batch)
            stu_stats = torch.vstack(stu_stats_list)
            eval_data_dict.update({
                'stu_stats': tensor2npy(stu_stats),
            })
        if hasattr(self.datatpl, 'Q_mat'):
            eval_data_dict.update({
                'Q_mat': tensor2npy(self.datatpl.Q_mat)
            })
        eval_result = {}
        for evaltpl in self.evaltpls: eval_result.update(
                evaltpl.eval(ignore_metrics=self.traintpl_cfg['ignore_metrics_in_train'], **eval_data_dict)
            )
        return eval_result

    @torch.no_grad()
    def inference(self, loader):
        self.model.eval()
        stu_id_list = list(range(len(loader)))
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            stu_id_list[idx] = batch_dict['stu_id']
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = eval_dict['y_gt'] if 'y_gt' in eval_dict else batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)
        stu_id = torch.hstack(stu_id_list)

        eval_data_dict = {
            'stu_id': stu_id,
            'y_pd': y_pd,
            'y_gt': y_gt,
        }
        if hasattr(self.model, 'get_stu_status'):
            stu_stats_list = []
            idx = torch.arange(0, self.datatpl_cfg['dt_info']['stu_count']).to(self.traintpl_cfg['device'])
            for i in range(0,self.datatpl_cfg['dt_info']['stu_count'], self.traintpl_cfg['eval_batch_size']):
                batch_stu_id = idx[i:i+self.traintpl_cfg['eval_batch_size']]
                batch = self.model.get_stu_status(batch_stu_id)
                stu_stats_list.append(batch)
            stu_stats = torch.vstack(stu_stats_list)
            eval_data_dict.update({
                'stu_stats': tensor2npy(stu_stats),
            })
        if hasattr(self.datatpl, 'Q_mat'):
            eval_data_dict.update({
                'Q_mat': tensor2npy(self.datatpl.Q_mat)
            })
        eval_result = {}
        for evaltpl in self.evaltpls: eval_result.update(evaltpl.eval(**eval_data_dict))
        return eval_result
