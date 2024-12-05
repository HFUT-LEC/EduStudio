from edustudio.traintpl import GeneralTrainTPL
from edustudio.utils.common import tensor2npy
import torch
from tqdm import tqdm
from edustudio.utils.callback import ModelCheckPoint, EarlyStopping, History, BaseLogger, Callback, CallbackList
from edustudio.utils.common import set_same_seeds
import shutil
import numpy as np

class DCDTrainTPL(GeneralTrainTPL):
    default_cfg = {}

    def one_fold_start(self, fold_id):
        self.logger.info(f"====== [FOLD ID]: {fold_id} ======")
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
            BaseLogger(self.logger, group_by_contains=['loss', 'user_', 'item_'])
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
        # set_same_seeds(self.traintpl_cfg['seed'])
        if type(self.datatpl.filling_Q_mat) is np.ndarray:
            self.model.Q_mat = torch.from_numpy(self.datatpl.filling_Q_mat).to(self.traintpl_cfg.device)
        else:
            self.model.Q_mat = self.datatpl.filling_Q_mat.to(self.traintpl_cfg.device)
        self.model.callback_list = self.callback_list
        if self.valid_loader is not None:
            self.fit(train_loader=self.train_loader, valid_loader=self.valid_loader)
        else:
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
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = eval_dict['y_gt'] if 'y_gt' in eval_dict else batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)

        eval_data_dict = {
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
        if hasattr(self.model, 'get_exer_emb'):
            eval_data_dict.update({
                'exer_emb': self.model.get_exer_emb(),
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
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = eval_dict['y_gt'] if 'y_gt' in eval_dict else batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)

        eval_data_dict = {
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

        if hasattr(self.model, 'get_exer_emb'):
            eval_data_dict.update({
                'exer_emb': self.model.get_exer_emb(),
            })

        if hasattr(self.datatpl, 'Q_mat'):
            eval_data_dict.update({
                'Q_mat': tensor2npy(self.datatpl.Q_mat)
            })
        eval_result = {}
        for evaltpl in self.evaltpls: eval_result.update(evaltpl.eval(**eval_data_dict))
        return eval_result
