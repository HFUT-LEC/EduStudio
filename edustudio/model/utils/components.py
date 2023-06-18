import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs

class ActivationUtil(object):
    @staticmethod
    def get_common_activation_layer(act_obj: Union[str, nn.Module] = "relu") -> nn.Module:
        if isinstance(act_obj, str):
            if act_obj.lower() == 'relu':
                return nn.ReLU(inplace=True)
            elif act_obj.lower() == 'sigmoid':
                return nn.Sigmoid()
            elif act_obj.lower() == 'linear':
                return Identity()
            elif act_obj.lower() == 'prelu':
                return nn.PReLU()
            elif act_obj.lower() == 'elu':
                return nn.ELU(inplace=True)
            elif act_obj.lower() == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
        else:
            return act_obj()


class PosMLP(nn.Module):
    """
        The Multi Layer Perceptron (MLP)
        note: output layer has no activation function, output layer has batch norm and dropout
    """

    def __init__(self, input_dim: int, output_dim: int, dnn_units: Union[list, tuple],
                 activation: Union[str, nn.Module, list] = 'relu', dropout_rate: float = 0.0,
                 use_bn: bool = False, device='cpu'):
        super().__init__()
        self.use_bn = use_bn
        dims_list = [input_dim] + list(dnn_units) + [output_dim]
        if type(activation) is list:
            assert len(activation) == len(dnn_units)

        self.linear_units_list = nn.ModuleList(
            [PosLinear(dims_list[i], dims_list[i + 1], bias=True) for i in range(len(dims_list) - 1)]
        )
        self.act_units_list = nn.ModuleList(
            [ActivationUtil.get_common_activation_layer(activation)] * len(dnn_units)
            if type(activation) is not list else [ActivationUtil.get_common_activation_layer(i) for i in activation]
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

        if use_bn is True:
            self.bn_units_list = nn.ModuleList(
                [nn.BatchNorm1d(dims_list[i + 1]) for i in range(len(dims_list) - 1)]
            )
            assert len(self.linear_units_list) == len(self.bn_units_list)
        assert len(self.linear_units_list) == len(self.act_units_list) + 1
        self.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp = input
        for i in range(len(self.act_units_list)):
            tmp = self.linear_units_list[i](tmp)
            if self.use_bn is True:
                tmp = self.bn_units_list[i](tmp)
            tmp = self.act_units_list[i](tmp)
            tmp = self.dropout_layer(tmp)
        tmp = self.linear_units_list[-1](tmp)
        if self.use_bn is True:
            tmp = self.bn_units_list[-1](tmp)
        output = tmp
        return output




class MLP(nn.Module):
    """
        The Multi Layer Perceptron (MLP)
        note: output layer has no activation function, output layer has batch norm and dropout
    """

    def __init__(self, input_dim: int, output_dim: int, dnn_units: Union[list, tuple],
                 activation: Union[str, nn.Module, list] = 'relu', dropout_rate: float = 0.0,
                 use_bn: bool = False, device='cpu'):
        super().__init__()
        self.use_bn = use_bn
        dims_list = [input_dim] + list(dnn_units) + [output_dim]
        if type(activation) is list:
            assert len(activation) == len(dnn_units)

        self.linear_units_list = nn.ModuleList(
            [nn.Linear(dims_list[i], dims_list[i + 1], bias=True) for i in range(len(dims_list) - 1)]
        )
        self.act_units_list = nn.ModuleList(
            [ActivationUtil.get_common_activation_layer(activation)] * len(dnn_units)
            if type(activation) is not list else [ActivationUtil.get_common_activation_layer(i) for i in activation]
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

        if use_bn is True:
            self.bn_units_list = nn.ModuleList(
                [nn.BatchNorm1d(dims_list[i + 1]) for i in range(len(dims_list) - 1)]
            )
            assert len(self.linear_units_list) == len(self.bn_units_list)
        assert len(self.linear_units_list) == len(self.act_units_list) + 1
        self.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp = input
        for i in range(len(self.act_units_list)):
            tmp = self.linear_units_list[i](tmp)
            if self.use_bn is True:
                tmp = self.bn_units_list[i](tmp)
            tmp = self.act_units_list[i](tmp)
            tmp = self.dropout_layer(tmp)
        tmp = self.linear_units_list[-1](tmp)
        if self.use_bn is True:
            tmp = self.bn_units_list[-1](tmp)
        output = tmp
        return output