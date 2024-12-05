from .general_traintpl import GeneralTrainTPL
import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class AdversarialTrainTPL(GeneralTrainTPL):
    default_cfg = {
        'lr': 0.001,
        'lr_d': 0.001,
        'g_rounds': 1,
        'd_rounds': 1,
        'optim': 'adam',
        'optim_d': 'adam',
    }

    def _get_optim(self, model_params, optimizer='adam', lr=0.001, weight_decay=0.0, eps=1e-8):
        """Get optimizer
        """
        if optimizer == "sgd":
            optim = torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "adam":
            optim = torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "adagrad":
            optim = torch.optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay, eps=eps)
        else:
            raise ValueError("unsupported optimizer")

        return optim

    def fit(self, train_loader, valid_loader):
        self.model.train()
        lr = self.traintpl_cfg['lr']
        lr_d = self.traintpl_cfg['lr_d']
        weight_decay = self.traintpl_cfg['weight_decay']
        eps = self.traintpl_cfg['eps']
        
        self.optimizer_g = self._get_optim(self.model.get_g_parameters(), self.traintpl_cfg['optim'], lr=lr, weight_decay=weight_decay, eps=eps)
        self.optimizer_d = self._get_optim(self.model.get_d_parameters(), self.traintpl_cfg['optim_d'], lr=lr_d, weight_decay=weight_decay, eps=eps)

        self.callback_list.on_train_begin()
        for epoch in range(self.traintpl_cfg['epoch_num']):
            self.callback_list.on_epoch_begin(epoch + 1)

            # train_for_generator
            g_rounds = self.traintpl_cfg['g_rounds']
            d_rounds = self.traintpl_cfg['d_rounds']
            logs = defaultdict(lambda: np.full((len(train_loader) * g_rounds,), np.nan, dtype=np.float32))
            for round_id in range(g_rounds):
                for batch_id, batch_dict in enumerate(
                        tqdm(train_loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[GEN:EPOCH={:03d}]".format(epoch + 1))
                ):
                    batch_dict = self.batch_dict2device(batch_dict)
                    loss_gen_dict = self.model.get_main_loss(**batch_dict)
                    loss_gen = torch.hstack([i for i in loss_gen_dict.values() if i is not None]).sum()
                    loss = loss_gen
                    self.optimizer_g.zero_grad()
                    loss.backward()
                    self.optimizer_g.step()
                    for k in loss_gen_dict: logs[k][batch_id + len(train_loader) * round_id] = loss_gen_dict[k].item() if loss_gen_dict[k] is not None else np.nan

            logs_g = {}
            for name in logs: logs_g[f"GEN_{name}"] = float(np.nanmean(logs[name]))

            # train_for_discriminator
            logs = defaultdict(lambda: np.full((len(train_loader) * d_rounds,), np.nan, dtype=np.float32))
            for round_id in range(d_rounds):
                for batch_id, batch_dict in enumerate(
                        tqdm(train_loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[DIS:EPOCH={:03d}]".format(epoch + 1))
                ):
                    batch_dict = self.batch_dict2device(batch_dict)
                    loss_pre_dict, loss_dis_dict = self.model.get_loss_dict(**batch_dict)
                    loss_pre = torch.hstack([i for i in loss_pre_dict.values() if i is not None]).sum()
                    loss_dis = torch.hstack([i for i in loss_dis_dict.values() if i is not None]).sum()
                    loss = loss_pre - loss_dis
                    self.optimizer_d.zero_grad()
                    loss.backward()
                    self.optimizer_d.step()
                    for k in loss_pre_dict: logs[k][batch_id + len(train_loader) * round_id] = loss_pre_dict[k].item() if loss_pre_dict[k] is not None else np.nan
                    for k in loss_dis_dict: logs[k][batch_id + len(train_loader) * round_id] = loss_dis_dict[k].item() if loss_dis_dict[k] is not None else np.nan

            logs_d = {}
            for name in logs: logs_d[f"DIS_{name}"] = float(np.nanmean(logs[name]))

            logs = logs_g
            logs.update(logs_d)
            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader)
                logs.update({f"{metric}": val_metrics[metric] for metric in val_metrics})

            self.callback_list.on_epoch_end(epoch + 1, logs=logs)
            if self.model.share_callback_dict.get('stop_training', False):
                break
            
        self.callback_list.on_train_end()
