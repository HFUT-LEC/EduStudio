from edustudio.traintpl import GeneralTrainTPL
from edustudio.utils.common import tensor2npy
import torch
from tqdm import tqdm


class GroupCDTrainTPL(GeneralTrainTPL):
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        stu_id_list = list(range(len(loader)))
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            stu_id_list[idx] = batch_dict['group_id']
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = eval_dict['y_gt'] if 'y_gt' in eval_dict else batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)
        group_id = torch.hstack(stu_id_list)

        eval_data_dict = {
            'group_id': group_id,
            'y_pd': y_pd,
            'y_gt': y_gt,
        }
        if hasattr(self.model, 'get_stu_status'):
            stu_stats_list = []
            idx = torch.arange(0, self.datatpl_cfg['dt_info']['group_count']).to(self.traintpl_cfg['device'])
            for i in range(0,self.datatpl_cfg['dt_info']['group_count'], self.traintpl_cfg['eval_batch_size']):
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
            stu_id_list[idx] = batch_dict['group_id']
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = eval_dict['y_gt'] if 'y_gt' in eval_dict else batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)
        group_id = torch.hstack(stu_id_list)

        eval_data_dict = {
            'group_id': group_id,
            'y_pd': y_pd,
            'y_gt': y_gt,
        }
        if hasattr(self.model, 'get_stu_status'):
            stu_stats_list = []
            idx = torch.arange(0, self.datatpl_cfg['dt_info']['group_count']).to(self.traintpl_cfg['device'])
            for i in range(0,self.datatpl_cfg['dt_info']['group_count'], self.traintpl_cfg['eval_batch_size']):
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
