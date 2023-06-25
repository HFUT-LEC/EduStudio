r"""
CDMFKC
##################################
Reference:
    Li et al. "Cognitive Diagnosis Focusing on Knowledge Concepts." in CIKM 2022.
"""
import torch
import torch.nn as nn
from ..utils.components import PosMLP
import torch.nn.functional as F
from ..gd_basemodel import GDBaseModel


class CDMFKC(GDBaseModel):
    """
    dnn_units: dimensions of the middle layers of a multilayer perceptron
    dropout_rate: dropout rate of a multilayer perceptron
    activation: activation function of a multilayer perceptron
    g_impact_a: hyperparameters of the original formula 5
    g_impact_b: hyperparameters of the original formula 5
    """
    default_cfg = {
        'dnn_units': [512, 256],
        'dropout_rate': 0.5,
        'activation': 'sigmoid',
        'g_impact_a': 0.5,
        'g_impact_b': 0.5
    }
    def __init__(self, cfg):
        """Pass parameters from other templates into the model

        Args:
            cfg (UnifyConfig): parameters from other templates
        """
        super().__init__(cfg)

    def build_cfg(self):
        """Initialize the parameters of the model"""
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']

    def add_extra_data(self, **kwargs):
        """Add the data required by the model to the self object from the data template"""
        self.Q_mat = kwargs['Q_mat'].to(self.device)

    def build_model(self):
        """Initialize the various components of the model"""
        self.student_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.k_difficulty = nn.Embedding(self.n_item, self.n_cpt)
        self.e_difficulty = nn.Embedding(self.n_item, 1)
        self.k_impact = nn.Embedding(self.n_item, self.n_cpt)
        self.pd_net = PosMLP(
            input_dim=self.n_cpt, output_dim=1, activation=self.modeltpl_cfg['activation'],
            dnn_units=self.modeltpl_cfg['dnn_units'], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )

    def forward(self, stu_id, exer_id, **kwargs):
        """Get the probability that the students will answer the exercise correctly

        Args:
            stu_id (torch.Tensor): Id of students. Shape of [inter_len]
            exer_id (torch.Tensor): Sequence of exercise id. Shape of [inter_len]

        Returns:
            torch.Tensor: prediction_tensor
        """
        # before prednet
        items_Q_mat = self.Q_mat[exer_id]  # Q_mat: exer_num * n_cpt; items_Q_mat: batch_exer_num * n_cpt
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)

        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_difficulty = torch.sigmoid(self.e_difficulty(exer_id))
        h_impact = torch.sigmoid(self.k_impact(exer_id))
        g_impact = torch.sigmoid(self.modeltpl_cfg['g_impact_a'] * h_impact + 
                                 self.modeltpl_cfg['g_impact_b'] * k_difficulty * e_difficulty)

        input_knowledge_point = items_Q_mat
        input_x = e_difficulty * (stat_emb + g_impact - k_difficulty) * input_knowledge_point

        pd = self.pd_net(input_x).sigmoid()

        return pd


    def get_main_loss(self, **kwargs):
        """Get the loss in the paper

        Returns:
            dict: {'loss_main': loss_value}
        """
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd = self(stu_id, exer_id).flatten()
        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        """

        Returns:
            dict: loss dict{'loss_main': loss_value}
        """
        return self.get_main_loss(**kwargs)
    
    @torch.no_grad()
    def predict(self, stu_id, exer_id,  **kwargs):
        """A function of get how well the model predicts students' responses to exercise questions

        Args:
            stu_id (torch.Tensor): Id of students.
            exer_id (torch.Tensor): Id of exercise.

        Returns:
            dict: {'y_pd':prediction_tensor}
        """
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }
    
    def get_stu_status(self, stu_id=None):
        """Get the Cognitive State of the Student

        Args:
            stu_id (torch.Tensor, optional): Id of students. Defaults to None.

        Returns:
            nn.Embedding: Cognitive State of the Student
        """
        if stu_id is not None:
            return self.student_emb(stu_id)
        else:
            return self.student_emb.weight
