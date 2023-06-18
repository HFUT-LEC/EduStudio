from .base_trainfmt import BaseTrainFmt
from edustudio.utils.common import UnifyConfig
import torch
from torch.utils.data import DataLoader

class GDTrainFmt(BaseTrainFmt):
    default_cfg = {
        'device': 'cuda:0',
        'seed': 2022,
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
    
    def _get_optim(self):
        optimizer = self.trainfmt_cfg['optim']
        lr = self.trainfmt_cfg['lr']
        weight_decay = self.trainfmt_cfg['weight_decay']
        eps = self.trainfmt_cfg['eps']
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

    # def batch_dict2device(self, batch_dict):
    #     return {
    #         k: v.to(self.trainfmt_cfg['device']) for k,v in batch_dict.items()
    #     }

    def start(self):
        super().start()
        self.build_loaders()
        self.model.build_cfg()
        extra_data = self.datafmt.get_extra_data() 
        if len(extra_data):
            self.model.add_extra_data(**extra_data)
        else:
            self.model.add_extra_data()
        for evalfmt in self.evalfmts:
            evalfmt.add_extra_data(**extra_data)
        self.model.build_model()
        self.model._init_params()
        self.model.to(self.model.device)

    def batch_dict2device(self, batch_dict):
        dic = {}
        for k, v in batch_dict.items():
            if not isinstance(v, list):
                dic[k] = v.to(self.trainfmt_cfg['device'])
            else:
                dic[k] = v
        return dic

    def build_loaders(self):
        batch_size = self.trainfmt_cfg['batch_size']
        num_workers = self.trainfmt_cfg['num_workers']
        eval_batch_size = self.trainfmt_cfg['eval_batch_size']
        if hasattr(self.datafmt, 'build_dataloaders'):
            train_loader, val_loader, test_loader = self.datafmt.build_dataloaders()
        else:
            train_dataset, val_dataset, test_dataset = self.datafmt.build_datasets()
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            if val_dataset is not None:
                val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader
