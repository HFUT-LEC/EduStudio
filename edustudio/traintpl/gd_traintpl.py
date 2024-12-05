from .base_traintpl import BaseTrainTPL
from edustudio.utils.common import UnifyConfig, set_same_seeds
import torch
from torch.utils.data import DataLoader
from edustudio.utils.callback import History
from collections import defaultdict
import numpy as np


class GDTrainTPL(BaseTrainTPL):
    default_cfg = {
        'device': 'cuda:0',
        'epoch_num': 100,
        'batch_size': 2048,
        'eval_batch_size': 2048,
        'num_workers': 0,

        'lr': 0.001,
        'optim': 'adam',
        'eps': 1e-8,
        'weight_decay': 0.0, 
    }

    def __init__(self, cfg: UnifyConfig):
        super().__init__(cfg)
        self.train_loader_list = []
        self.valid_loader_list = []
        self.test_loader_list = []
    
    def _get_optim(self, optimizer='adam', lr=0.001, weight_decay=0.0, eps=1e-8):
        """Get optimizer
        """
        if optimizer == "sgd":
            optim = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "adagrad":
            optim = torch.optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        else:
            raise ValueError("unsupported optimizer")

        return optim

    def start(self):
        """entrypoint of starting a training process
        """
        super().start()
        self.build_loaders()

        best_metric_value_dict = defaultdict(list)

        for fold_id in range(self.datatpl_cfg['n_folds']):
            self.train_loader = self.train_loader_list[fold_id]
            self.train_loader.dataset.set_info_for_fold(fold_id)
            self.datatpl.set_info_for_fold(fold_id)

            self.valid_loader = None
            if self.datatpl.hasValidDataset:
                self.valid_loader = self.valid_loader_list[fold_id]
                self.valid_loader.dataset.set_info_for_fold(fold_id)
            self.test_loader = self.test_loader_list[fold_id]
            self.test_loader.dataset.set_info_for_fold(fold_id)

            extra_data = self.datatpl.get_extra_data() 

            set_same_seeds(self.traintpl_cfg['seed'])
            self.model = self.get_model_obj()
            self.model.build_cfg()
            self.model.add_extra_data(**extra_data)
            for evaltpl in self.evaltpls:
                evaltpl.add_extra_data(**extra_data)
            self.model.build_model()
            self.model._init_params()
            self.model.to(self.model.device)


            metrics = self.one_fold_start(fold_id)
            for m in metrics:
                best_metric_value_dict[m].append(metrics[m])
            self.logger.info("=="*10)
        
        best_metric_value_dict_mean = {}
        for metric_name, val_list in best_metric_value_dict.items():
            best_metric_value_dict_mean[metric_name] = np.mean(val_list)
            self.logger.info(f"All Fold Mean {metric_name} = {best_metric_value_dict_mean[metric_name]}")

        History.dump_json(best_metric_value_dict, f"{self.frame_cfg.temp_folder_path}/result-list.json")
        History.dump_json(best_metric_value_dict_mean, f"{self.frame_cfg.temp_folder_path}/result.json")
    
    def one_fold_start(self, fold_id):
        """training process of one one fold

        Args:
            fold_id (int): fold id
        """
        self.logger.info(f"====== [FOLD ID]: {fold_id} ======")

    def batch_dict2device(self, batch_dict):
        dic = {}
        for k, v in batch_dict.items():
            if not isinstance(v, list):
                dic[k] = v.to(self.traintpl_cfg['device'])
            else:
                dic[k] = v
        return dic

    def build_loaders(self):
        """build dataloaders
        """
        batch_size = self.traintpl_cfg['batch_size']
        num_workers = self.traintpl_cfg['num_workers']
        eval_batch_size = self.traintpl_cfg['eval_batch_size']
        if hasattr(self.datatpl, 'build_dataloaders'):
            self.train_loader_list, self.valid_loader_list, self.test_loader_list = self.datatpl.build_dataloaders()
        else:
            train_dt_list, valid_dt_list, test_dt_list = self.datatpl.build_datasets()
            for fid in range(self.datatpl_cfg['n_folds']):
                train_loader = DataLoader(dataset=train_dt_list[fid], shuffle=True, batch_size=batch_size, 
                                          num_workers=num_workers, collate_fn=train_dt_list[fid].collate_fn)
                self.train_loader_list.append(train_loader)
                if self.hasValidDataset:
                    valid_loader = DataLoader(dataset=valid_dt_list[fid], shuffle=False, batch_size=eval_batch_size,
                                               num_workers=num_workers, collate_fn=valid_dt_list[fid].collate_fn)
                    self.valid_loader_list.append(valid_loader)
                test_loader = DataLoader(dataset=test_dt_list[fid], shuffle=False, batch_size=eval_batch_size, 
                                         num_workers=num_workers, collate_fn=test_dt_list[fid].collate_fn)
                self.test_loader_list.append(test_loader)
