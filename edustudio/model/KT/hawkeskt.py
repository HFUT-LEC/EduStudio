r"""
CT_NCM
##################################
Reference:
    Chenyang Wang et al. "Temporal cross-effects in knowledge tracing." in WSDM 2021.
"""
import numpy as np
from ..gd_basemodel import GDBaseModel
import torch
import torch.nn.functional as F


class HawkesKT(GDBaseModel):
    default_cfg = {
        'dim_s': 50,  # 序列长度
        'emb_size': 64,
        'time_log': 5,  # Log base of time intervals.
    }

    def __init__(self, cfg):
        """Pass parameters from other templates into the model

        Args:
            cfg (UnifyConfig): parameters from other templates
        """
        super().__init__(cfg)

    def build_cfg(self):
        """Initialize the parameters of the model"""
        self.problem_num = self.datatpl_cfg['dt_info']['exer_count']
        self.skill_num = self.datatpl_cfg['dt_info']['cpt_count']
        self.emb_size = self.modeltpl_cfg['emb_size']
        self.time_log = self.modeltpl_cfg['time_log']

    def build_model(self):
        """Initialize the various components of the model"""
        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)  # Corresponding to the PA in the paper
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

    def _init_params(self):
        """Parameter initialization of each component of the model"""
        torch.nn.init.normal_(self.problem_base.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.skill_base.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.alpha_inter_embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.alpha_skill_embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.beta_inter_embeddings.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.beta_skill_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, exer_seq, start_timestamp_seq, cpt_unfold_seq, **kwargs):
        """A function of how well the model predicts students' responses to exercise questions

        Args:
            exer_seq (torch.Tensor): Sequence of exercise id. Shape of [batch_size, seq_len]
            start_timestamp_seq (torch.Tensor): The time the student started answering the question. Shape of [batch_size, seq_len]
            cpt_unfold_seq (torch.Tensor): Knowledge concepts corresponding to exercises. Shape of [batch_size, seq_len]

        Returns:
            torch.Tensor: The predictions of the model
        """
        skills = cpt_unfold_seq     # [batch_size, seq_len] One exercise corresponds to one knowledge point
        problems = exer_seq  # [batch_size, seq_len] sequence of batch_size students
        # time = [i for i in range(start_timestamp_seq.shape[1])]
        # times = torch.Tensor([time for i in range(start_timestamp_seq.shape[0])])
        times = start_timestamp_seq - start_timestamp_seq[:,[0]]        # [batch_size, seq_len]

        mask_labels = kwargs['mask_seq'].long()
        inters = skills + mask_labels * self.skill_num

        alpha_src_emb = self.alpha_inter_embeddings(inters)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        beta_src_emb = self.beta_inter_embeddings(inters)  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(skills)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)  

        delta_t = (times[:, :, None] - times[:, None, :]).abs().double()  # Get the absolute value of the time at different time steps
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)  # The cross_effects of the paper (4)

        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.cuda() if self.device != 'cpu' else mask
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(skills).squeeze(dim=-1)

        prediction = (problem_bias + skill_bias + sum_t).sigmoid()
        return prediction

    @torch.no_grad()
    def predict(self, **kwargs):
        """A function of get how well the model predicts students' responses to exercise questions and the groundtruth

        Returns:
            dict: The predictions of the model and the real situation
        """
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1]
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
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd.double(), target=y_gt.double()
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
