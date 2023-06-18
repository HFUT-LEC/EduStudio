import torch.nn as nn
import torch
from abc import abstractmethod
from .utils.common import xavier_normal_initialization, xavier_uniform_initialization, kaiming_normal_initialization, kaiming_uniform_initialization
from .basemodel import BaseModel, BaseProxyModel

class GDBaseModel(BaseModel):
    default_cfg = {
        'param_init_type': 'xavier_normal',
        'pretrained_file_path': "",
    }
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = self.trainfmt_cfg['device']
        self.share_callback_dict = {
            "stop_training": False
        }

    @abstractmethod
    def build_cfg(self):
        """
            construct model config
        """
        pass

    @abstractmethod
    def build_model(self):
        """
            construct model component
        """
        pass


    def _init_params(self):
        if self.model_cfg['param_init_type'] == 'default':
            pass
        elif self.model_cfg['param_init_type'] == 'xavier_normal':
            self.apply(xavier_normal_initialization)
        elif self.model_cfg['param_init_type'] == 'xavier_uniform':
            self.apply(xavier_uniform_initialization)
        elif self.model_cfg['param_init_type'] == 'kaiming_normal':
            self.apply(kaiming_normal_initialization)
        elif self.model_cfg['param_init_type'] == 'kaiming_uniform':
            self.apply(kaiming_uniform_initialization)
        elif self.model_cfg['param_init_type'] == 'init_from_pretrained':
            self._load_params_from_pretrained()

    def _load_params_from_pretrained(self):
        self.load_state_dict(torch.load(self.model_cfg['pretrained_file_path']))

    def predict(self, **kwargs):
        pass

    def get_loss_dict(self, **kwargs):
        pass

