from ..gd_basemodel import GDBaseModel

import torch.nn as nn
import torch
import torch.nn.functional as F


class DKTDSC(GDBaseModel):
    default_cfg = {
        'emb_size': 200,
        'hidden_size': 200,
        'num_layers': 1,
        'dropout_rate': 0.5,
        'rnn_or_lstm': 'lstm',
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    # def add_extra_data(self, **kwargs):
    #     self.cluster = kwargs.pop('cluster')

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_clusters = self.datatpl_cfg['dt_info']['n_cluster']

    def build_model(self):
        self.exer_emb = nn.Embedding(
            self.n_item * 2, self.modeltpl_cfg['emb_size']
        )
        self.next_id_emb = nn.Embedding(
            self.n_item, self.modeltpl_cfg['emb_size']
        )
        if self.modeltpl_cfg['rnn_or_lstm'] == 'rnn':
            self.seq_model = nn.RNN(
                self.modeltpl_cfg['emb_size'] * 2, self.modeltpl_cfg['hidden_size'],
                self.modeltpl_cfg['num_layers'], batch_first=True
            )
        else:
            self.seq_model = nn.LSTM(
                self.modeltpl_cfg['emb_size'] * 2, self.modeltpl_cfg['hidden_size'],
                self.modeltpl_cfg['num_layers'], batch_first=True
            )
        self.dropout_layer = nn.Dropout(self.modeltpl_cfg['dropout_rate'])
        self.fc_layer = nn.Linear(self.modeltpl_cfg['hidden_size'], self.n_clusters + 1)


    def forward(self, exer_seq, label_seq, **kwargs):
        seg_seq = kwargs['seg_seq']
        cluster = kwargs['cluster']

        zeros = torch.zeros_like(cluster, device=self.device)
        cluster_update = ~(seg_seq == 0) * (cluster + 1) + (seg_seq == 0) * zeros

        input_x = self.exer_emb(exer_seq[:,:-1] + label_seq[:,:-1].long() * self.n_item)
        next_id = self.next_id_emb(exer_seq[:,1:])

        input = torch.cat((input_x, next_id), dim=2)

        output, _ = self.seq_model(input)
        output = self.dropout_layer(output)
        y_pd = self.fc_layer(output).sigmoid()
        y_pd = torch.gather(y_pd, dim=2, index=cluster_update[:,1:].unsqueeze(2)).squeeze()
        return y_pd

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        # y_pd = y_pd[:, :-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def get_main_loss(self, **kwargs):
        y_pd = self(**kwargs)
        # y_pd = y_pd[:, :-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)