#  Customize Model Template

Here, we present how to develop a new Model Template, and apply it into EduStudio. EduStudio provides the Model Protocol in ``EduStudio.edustudio.model.basemodeltpl.BaseModelTPL`` (``BaseModelTPL``).  

## Protocol 

### BaseModelTPL

The protocols in ``BaseModelTPL`` are listed as follows.  

| name           | description                                    | type               | note                    |
| -------------- | ---------------------------------------------- | ------------------ | ----------------------- |
| default_cfg    | the default configuration                      | class variable     |                         |
| add_extra_data | add extra data in addition to the data loaders | function interface | implemented by subclass |

###  BaseProxyModelTPL

 This protocol is for some works proposed general frameworks which can replace the backbone model. We implement the ``BaseProxyModelTPL`` Protocol. Users only need to inherit this protocol and set the ``backbone_model_cls`` parameter in the ``default_cfg``.   

## ModelTPL

EduStudio provides the ModelTPL in ``EduStudio.edustudio.model.gd_basemodeltpl.GDBaseModelTPL`` (``GDBaseModelTPL``).  ``GDBaseModelTPL`` inherents ``BaseModelTPL``, and the methods in ``GDBaseModelTPL`` are listed as follows. 

| name                         | description               | type               | note                    |
| ---------------------------- | ------------------------- | ------------------ | ----------------------- |
| default_cfg                  | the default configuration | class variable     |                         |
| build_cfg                    | construct model config    | abstract method    | implemented by subclass |
| build_model                  | construct model component | abstract method    | implemented by subclass |
| predict                      | predict function          | function interface | implemented by subclass |
| get_loss_dict                | obtain loss               | function interface | implemented by subclass |
| _init_params                 | initial parameters        | function interface | implemented by subclass |
| _load_params_from_pretrained | load parameters as dict   | function interface | implemented by subclass |

##  Develop a New ModelTPL in EduStudio

When you develope a new model in EduStudio, then you can inherent ``GDBaseModelTPL`` and implement the abstract methods ``build_cfg()`` and ``build_model()``. Then, you can revise the function ``predict()`` and ``get_loss_dict()``. You can also define the configuration of the new model template in the dictionary ``default_cfg``.  



If you want to develope a new ModelTPL for a backbone-style model, then you can inherent ``BaseProxyModelTPL`` and set the ``backbone_model_cls`` parameter in ``default_cfg``. Then, you can implement the some methods such as ``build_model()``, ``get_loss_dict()``, and ``predict()``. 



###  Example 1: Develop a traditional ModelTPL

```python
from ..gd_basemodeltpl import GDBaseModelTPL

class NewModelTPL(GDBaseModelTPL):
    default_cfg = {
    'dnn_units': [512, 256],
    'dropout_rate': 0.5,
    'activation': 'sigmoid',
    'disc_scale': 10
}

def __init__(self, cfg):
    super().__init__(cfg)

def build_cfg(self):
    self.n_user = self.datafmt_cfg['dt_info']['stu_count']
    ...

def build_model(self):
    ...
    self.pd_net = PosMLP(
        input_dim=self.n_cpt, output_dim=1, activation=self.model_cfg['activation'],
        dnn_units=self.model_cfg['dnn_units'], dropout_rate=self.model_cfg['dropout_rate']
    )

def forward(self, stu_id, exer_id, Q_mat, **kwargs):
    ...
    pd = self.pd_net(input_x).sigmoid()
    return pd

@torch.no_grad()
def predict(self, stu_id, exer_id, Q_mat, **kwargs):
    return {
        'y_pd': self(stu_id, exer_id, Q_mat).flatten(),
    }

def get_main_loss(self, **kwargs):
    ...
    pd = self(stu_id, exer_id, Q_mat).flatten()
    loss = F.binary_cross_entropy(input=pd, target=label)
    return {
        'loss_main': loss
    }

def get_loss_dict(self, **kwargs):
    return self.get_main_loss(**kwargs)
```


### Example 2: Develop a backbone-style ModelTPL

```python
from ..basemodeltpl import BaseProxyModelTPL

class NewBackboneModelTPL(BaseProxyModelTPL):
    default_cfg = {
    "backbone_model_cls": "IRT",
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        super().build_model()
        self.irr_pair_loss = PairSCELoss()

    def get_main_loss(self, **kwargs):
        pair_exer = kwargs['pair_exer']
        pair_pos_stu = kwargs['pair_pos_stu']
        pair_neg_stu = kwargs['pair_neg_stu']

        kwargs['exer_id'] = pair_exer
        kwargs['stu_id'] = pair_pos_stu
        pos_pd = self(**kwargs).flatten()
        kwargs['stu_id'] = pair_neg_stu
        neg_pd = self(**kwargs).flatten()

        return {
            'loss_main': self.irr_pair_loss(pos_pd, neg_pd)
        }

class PairSCELoss(nn.Module):
    ...
```
