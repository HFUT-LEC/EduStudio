import torch.nn as nn
import torch
from abc import abstractmethod
from .utils.common import xavier_normal_initialization, xavier_uniform_initialization, kaiming_normal_initialization, kaiming_uniform_initialization
from .basemodel import BaseModel


class GDBaseModel(BaseModel):
    """
    The model that using gradient descent method can inherit the class
    """
    default_cfg = {
        'param_init_type': 'xavier_normal', # initialization method of model paramters
        'pretrained_file_path': "", # file path of pretrained model parameters
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = self.traintpl_cfg['device']
        self.share_callback_dict = {
            "stop_training": False
        }

    @abstractmethod
    def build_cfg(self):
        """Construct model config
        """
        pass

    @abstractmethod
    def build_model(self):
        """Construct model component
        """
        pass

    def _init_params(self):
        """Initialize the model parameters
        """
        if self.modeltpl_cfg['param_init_type'] == 'default':
            pass
        elif self.modeltpl_cfg['param_init_type'] == 'xavier_normal':
            self.apply(xavier_normal_initialization)
        elif self.modeltpl_cfg['param_init_type'] == 'xavier_uniform':
            self.apply(xavier_uniform_initialization)
        elif self.modeltpl_cfg['param_init_type'] == 'kaiming_normal':
            self.apply(kaiming_normal_initialization)
        elif self.modeltpl_cfg['param_init_type'] == 'kaiming_uniform':
            self.apply(kaiming_uniform_initialization)
        elif self.modeltpl_cfg['param_init_type'] == 'init_from_pretrained':
            self._load_params_from_pretrained()

    def _load_params_from_pretrained(self):
        """Load pretrained model parameters
        """
        self.load_state_dict(torch.load(self.modeltpl_cfg['pretrained_file_path']))

    def predict(self, **kwargs):
        """predict process
        """
        pass

    def get_loss_dict(self, **kwargs):
        """Get a dict object. The key is the loss name, the value is the loss
        """
        pass
