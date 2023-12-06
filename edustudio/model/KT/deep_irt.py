r"""
DEEPIRT
##################################
Reference:
    Yeung et al. "Deep-IRT: Make deep learning based knowledge tracing explainable using item response theory." in EDM 2019.
"""
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepIRT(GDBaseModel):
    """
    dim_s: sequence length
    size_m: embedding size of a key memory slot
    drop_out: dropout ratio before prediction
    """
    default_cfg = {
        'dim_s': 200,
        'size_m': 50,
        'drop_out': 0.2,
    }

    def __init__(self, cfg):
        """Pass parameters from other templates into the model

        Args:
            cfg (UnifyConfig): parameters from other templates
        """
        super().__init__(cfg)

    def _init_params(self):
        """Parameter initialization of each component of the model"""
        super()._init_params()
        nn.init.kaiming_normal_(self.Mk)  # 使用正态分布对输入张量进行赋值
        nn.init.kaiming_normal_(self.Mv0)


    def build_cfg(self):
        """Initialize the parameters of the model"""
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']

    def build_model(self):
        """Initialize the various components of the model"""
        self.k_emb_layer = nn.Embedding(self.n_item, self.modeltpl_cfg['dim_s'])
        self.Mk = nn.Parameter(torch.Tensor(self.modeltpl_cfg['size_m'], self.modeltpl_cfg['dim_s']))
        self.Mv0 = nn.Parameter(torch.Tensor(self.modeltpl_cfg['size_m'], self.modeltpl_cfg['dim_s']))

        self.v_emb_layer = nn.Embedding(self.n_item * 2, self.modeltpl_cfg['dim_s'])

        self.f_layer = nn.Linear(self.modeltpl_cfg['dim_s'] * 2, self.modeltpl_cfg['dim_s'])
        self.theta_layer = nn.Linear(self.modeltpl_cfg['dim_s'], self.modeltpl_cfg['dim_s'])
        self.beta_layer = nn.Linear(self.modeltpl_cfg['dim_s'], self.modeltpl_cfg['dim_s'])
        self.dropout_layer = nn.Dropout(self.modeltpl_cfg['drop_out'])
        self.p_layer = nn.Linear(self.modeltpl_cfg['dim_s'], 1)

        self.e_layer = nn.Linear(self.modeltpl_cfg['dim_s'], self.modeltpl_cfg['dim_s'])
        self.a_layer = nn.Linear(self.modeltpl_cfg['dim_s'], self.modeltpl_cfg['dim_s'])


    def forward(self, exer_seq, label_seq, **kwargs):
        """A function of how well the model predicts students' responses to exercise questions

        Args:
            exer_seq (torch.Tensor): Sequence of exercise id. Shape of [batch_size, seq_len]
            label_seq (torch.Tensor): Sequence of students' answers to exercises. Shape of [batch_size, seq_len]

        Returns:
            torch.Tensor: The predictions of the model
        """
        batch_size = exer_seq.shape[0]
        x = exer_seq + self.n_item * label_seq
        k = self.k_emb_layer(exer_seq.long())
        v = self.v_emb_layer(x.long())

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        theta = torch.tanh(self.theta_layer(f))
        beta = torch.tanh(self.beta_layer(k))

        p = self.p_layer(self.dropout_layer(3*theta - beta))

        y_pd = torch.sigmoid(p)

        return y_pd.squeeze(-1)

    @torch.no_grad()
    def predict(self, **kwargs):
        """A function of get how well the model predicts students' responses to exercise questions and the groundtruth

        Returns:
            dict: The predictions of the model and the real situation
        """
        y_pd = self(**kwargs)
        y_pd = y_pd[:, 1:]
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
        """

        Returns:
            dict: {'loss_main': loss_value}
        """
        y_pd = self(**kwargs)
        y_pd = y_pd[:, 1:]
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
        """

        Returns:
            dict: {'loss_main': loss_value}
        """
        return self.get_main_loss(**kwargs)
