import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, constant_


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_
    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def kaiming_normal_initialization(module):
    r""" using `kaiming_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`kaiming_normal`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_normal_#torch.nn.init.kaiming_normal_
    Examples:
        >>> self.apply(kaiming_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        kaiming_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        kaiming_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def kaiming_uniform_initialization(module):
    r""" using `kaiming_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`kaiming_uniform`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_uniform_#torch.nn.init.kaiming_uniform_
    Examples:
        >>> self.apply(kaiming_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        kaiming_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        kaiming_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
