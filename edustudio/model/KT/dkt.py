r"""
DKT
##########################################

Reference:
    Chris Piech et al. "Deep knowledge tracing" in NIPS 2015.

"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class DKT(GDBaseModel):
    default_cfg = {
        'emb_size': 100,
        'hidden_size': 100,
        'num_layers': 1,
        'dropout_rate': 0.2,
        'rnn_or_lstm': 'lstm',
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        assert self.modeltpl_cfg['rnn_or_lstm'] in {'rnn', 'lstm'}

    def build_model(self):
        self.exer_emb = nn.Embedding(
            self.n_item * 2, self.modeltpl_cfg['emb_size']
        )
        if self.modeltpl_cfg['rnn_or_lstm'] == 'rnn':
            self.seq_model = nn.RNN(
                self.modeltpl_cfg['emb_size'], self.modeltpl_cfg['hidden_size'], 
                self.modeltpl_cfg['num_layers'], batch_first=True
            )
        else:
            self.seq_model = nn.LSTM(
                self.modeltpl_cfg['emb_size'], self.modeltpl_cfg['hidden_size'], 
                self.modeltpl_cfg['num_layers'], batch_first=True
            )
        self.dropout_layer = nn.Dropout(self.modeltpl_cfg['dropout_rate'])
        self.fc_layer = nn.Linear(self.modeltpl_cfg['hidden_size'], self.n_item)

    def forward(self, exer_seq, label_seq, **kwargs):
        input_x = self.exer_emb(exer_seq + label_seq.long() * self.n_item)
        output, _ = self.seq_model(input_x)
        output = self.dropout_layer(output)
        y_pd = self.fc_layer(output).sigmoid()
        return y_pd
    
    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
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
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
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
