r"""
DKTForget
##########################################

Reference:
    Koki Nagatani et al. "Augmenting Knowledge Tracing by Considering Forgetting Behavior" in WWW 2019.

Reference Code:
    https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/dkt_forget.py

"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class DKTForget(GDBaseModel):
    default_cfg = {
        'emb_size': 100,
        'num_layers': 1,
        'dropout_rate': 0.2,
        'rnn_or_lstm': 'lstm',
        'integration_type': 'concat_multiply'
    }

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_rgap = self.datatpl_cfg['dt_info']['n_rgap']
        self.n_sgap = self.datatpl_cfg['dt_info']['n_sgap']
        self.n_pcount = self.datatpl_cfg['dt_info']['n_pcount']
        assert self.modeltpl_cfg['rnn_or_lstm'] in {'rnn', 'lstm'}
        
    def build_model(self):
        self.exer_emb = nn.Embedding(
            self.n_item * 2, self.modeltpl_cfg['emb_size']
        )
        self.integration_compont = IntegrationComponent(
            n_rgap=self.n_rgap, n_sgap=self.n_sgap, n_pcount=self.n_pcount,
            emb_dim=self.modeltpl_cfg['emb_size'], device=self.device,
            integration_type=self.modeltpl_cfg['integration_type']
        )
        if self.modeltpl_cfg['rnn_or_lstm'] == 'rnn':
            self.seq_model = nn.RNN(
                self.integration_compont.output_dim, self.modeltpl_cfg['emb_size'], 
                self.modeltpl_cfg['num_layers'], batch_first=True
            )
        else:
            self.seq_model = nn.LSTM(
                self.integration_compont.output_dim, self.modeltpl_cfg['emb_size'], 
                self.modeltpl_cfg['num_layers'], batch_first=True
            )
        self.dropout_layer = nn.Dropout(self.modeltpl_cfg['dropout_rate'])
        self.fc_layer = nn.Linear(self.integration_compont.output_dim, self.n_item)

    def forward(self, exer_seq, label_seq, r_gap, s_gap, p_count, **kwargs):
        input_x = self.exer_emb(exer_seq + label_seq.long() * self.n_item)
        input_x = self.integration_compont(input_x, r_gap, s_gap, p_count)
        h, _ = self.seq_model(input_x)
        output = self.integration_compont(h, r_gap, s_gap, p_count)
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


class IntegrationComponent(nn.Module):
    def __init__(self, n_rgap, n_sgap, n_pcount, emb_dim, integration_type="concat_multiply", device="cpu") -> None:
        super().__init__()
        self.rgap_eye = torch.eye(n_rgap).to(device)
        self.sgap_eye = torch.eye(n_sgap).to(device)
        self.pcount_eye = torch.eye(n_pcount).to(device)
        n_total = n_rgap + n_sgap + n_pcount

        if integration_type == 'concat':
            self.integration_func = self.concat
            self.output_dim = emb_dim + n_total
        elif integration_type == 'multiply':
            self.C = nn.Linear(n_total, emb_dim, bias=False)
            self.integration_func = self.multiply
            self.output_dim = emb_dim
        elif integration_type == 'concat_multiply':
            self.C = nn.Linear(n_total, emb_dim, bias=False)
            self.integration_func = self.concat_multiply
            self.output_dim = emb_dim + n_total
        elif integration_type == 'bi_interaction':
            self.C = nn.Linear(n_total, emb_dim, bias=False)
            self.integration_func = self.bi_interaction
            pass
        else:
            raise ValueError(f"unknown intefration_type: {integration_type}")

    def forward(self, v_t, r_gap, s_gap, p_count):
        return self.integration_func(v_t,r_gap, s_gap, p_count)

    def concat(self, v_t, r_gap, s_gap, p_count):
        rgap, sgap, pcount = self.rgap_eye[r_gap], self.sgap_eye[s_gap], self.pcount_eye[p_count]
        c_t = torch.cat((rgap, sgap, pcount), -1)
        return torch.cat((v_t, c_t), -1)

    def multiply(self, v_t, r_gap, s_gap, p_count):
        rgap, sgap, pcount = self.rgap_eye[r_gap], self.sgap_eye[s_gap], self.pcount_eye[p_count]
        c_t = torch.cat((rgap, sgap, pcount), -1)
        Cct = self.C(c_t)
        theta = torch.mul(v_t, Cct)
        return theta

    def concat_multiply(self, v_t, r_gap, s_gap, p_count):
        rgap = self.rgap_eye[r_gap]
        sgap = self.sgap_eye[s_gap]
        pcount = self.pcount_eye[p_count]
        c_t = torch.cat((rgap, sgap, pcount), -1)
        Cct = self.C(c_t)
        theta = torch.mul(v_t, Cct)
        theta = torch.cat((theta, c_t), -1)
        return theta

    def bi_interaction(self, v_t, r_gap, s_gap, p_count):
        raise NotImplementedError
