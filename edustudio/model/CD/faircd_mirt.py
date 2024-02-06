from edustudio.model.CD import MIRT
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


class FairCD_MIRT(MIRT):
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
        self.bias_theta = nn.Embedding(self.n_user, self.emb_dim)
        self.discriminator = self.disc_cls(
                    input_dim=self.emb_dim,
                    output_dim=self.out_dim,
                    device=self.device
               )
        self.predictor = self.pred_cls(
                    input_dim=self.emb_dim,
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
        x = self.theta(stu_id)
        loss_adv_dict = {
            f'loss_dis_{self.attr_name}': self.discriminator(x, self.label[stu_id])
        }

        return loss_adv_dict

    def get_pre_loss(self, **kwargs):
        stu_id = kwargs['stu_id'].unique()
        x = self.bias_theta(stu_id)
        loss_adv_dict = {
            f'loss_dis_{self.attr_name}': self.predictor(x, self.label[stu_id])
        }

        return loss_adv_dict

    def get_loss_dict(self, **kwargs):
        loss_pre= self.get_pre_loss(**kwargs)
        loss_dis = self.get_adv_loss(**kwargs)

        return loss_pre, loss_dis

    def forward(self, stu_id, exer_id, **kwargs):
        theta = self.theta(stu_id) + self.bias_theta(stu_id)
        a = self.a(exer_id)
        b = self.b(exer_id).flatten()

        if self.modeltpl_cfg['a_range'] is not None:
            a = self.modeltpl_cfg['a_range'] * torch.sigmoid(a)
        else:
            a = F.softplus(a) # 让区分度大于0，保持单调性假设
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The diff_range or a_range is too large.')
        return self.irf(theta, a, b)

