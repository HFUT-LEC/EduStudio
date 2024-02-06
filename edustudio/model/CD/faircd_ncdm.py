from edustudio.model.CD import NCDM
from edustudio.model.utils.components import MLP
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.preprocessing import LabelEncoder



class DiscriminatorForDiscreteSingleAttr(nn.Module):  
    """  
    判别器，仅针对单值离散属性  
    """  
    def __init__(self, input_dim, output_dim, activation='relu', device='cuda:0'):  
        super().__init__()  
        dnn_units = [int(input_dim/2)]  
        self.mlp = MLP(  
        input_dim=input_dim,  
        output_dim=output_dim,  
        dnn_units=dnn_units,  
        activation=activation,  
        device=device  
        )  
        self.loss = nn.CrossEntropyLoss()  
        self.to(device)

    def forward(self, x, y):  
        """_summary_  
        Args:  
        x (_type_): BatchNum x FeatNum  
        y (_type_): (BatachNum,)  
        """  
        pd = self.mlp(x)  
        return self.loss(pd, y.flatten()) 


class FairCD_NCDM(NCDM):
    default_cfg = {
        'sensi_attr': 'gender:token'
    }

    def add_extra_data(self, **kwargs):
        super().add_extra_data(**kwargs)
        self.df_stu = kwargs['df_stu']
        self.df_stu_index= self.df_stu.set_index("stu_id:token")
        assert self.df_stu['stu_id:token'].max() + 1 == kwargs['dt_info']['stu_count']

        self.attr_name = self.modeltpl_cfg['sensi_attr']
        attr_sufix = self.attr_name.split(":")[-1]

        if attr_sufix == 'token':
            self.disc_cls = DiscriminatorForDiscreteSingleAttr
            self.pred_cls = DiscriminatorForDiscreteSingleAttr
            n_classes = self.df_stu[self.attr_name].nunique()
            self.out_dim = n_classes
            lbe = LabelEncoder()
            lbe.fit(self.df_stu[self.attr_name])
            self.label = torch.LongTensor(
                [lbe.transform([self.df_stu_index[self.attr_name].loc[sid]])[-1] for sid in range(kwargs['dt_info']['stu_count'])]
            ).to(self.device)
        else:
            raise NotImplementedError

    def build_model(self):
        super().build_model()
        self.bias_student_emb = nn.Embedding(self.n_user, self.n_cpt)
        self.discriminator = self.disc_cls(
                    input_dim=self.n_cpt,
                    output_dim=self.out_dim,
                    device=self.device
               )
        self.predictor = self.pred_cls(
                    input_dim=self.n_cpt,
                    output_dim=self.out_dim,
                    device=self.device
               )

    def get_g_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'discriminator' not in name and 'predictor' not in name:
                yield param

    def get_d_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'discriminator' in name or 'predictor' in name:
                yield param

    def get_adv_loss(self, **kwargs):
        stu_id = kwargs['stu_id'].unique()
        x = self.student_emb(stu_id)
        loss_adv_dict = {
            f'loss_dis_{self.attr_name}': self.discriminator(x, self.label[stu_id])
        }

        return loss_adv_dict

    def get_pre_loss(self, **kwargs):
        stu_id = kwargs['stu_id'].unique()
        x = self.bias_student_emb(stu_id)
        loss_adv_dict = {
            f'loss_dis_{self.attr_name}': self.predictor(x, self.label[stu_id])
        }

        return loss_adv_dict

    def get_loss_dict(self, **kwargs):
        loss_pre= self.get_pre_loss(**kwargs)
        loss_dis = self.get_adv_loss(**kwargs)

        return loss_pre, loss_dis

    def forward(self, stu_id, exer_id, **kwargs):
        # before prednet
        stu_emb = self.student_emb(stu_id) + self.bias_student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_difficulty = torch.sigmoid(self.e_difficulty(exer_id)) * self.modeltpl_cfg['disc_scale']
        # prednet
        input_knowledge_point = self.Q_mat[exer_id]
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        pd = self.pd_net(input_x).sigmoid()
        return pd

