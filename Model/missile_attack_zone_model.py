import torch
import torch.nn as nn
from Model.mlp.mlp_model import mlp


class MissileHitProb(nn.Module):
    """
    多层感知机 导弹攻击区命中概率拟合模型
    """

    def __init__(self, input_dim, hidden_sizes, activation):
        """
        :param input_dim: 输入维度
        :param hidden_sizes: 元素为每层隐藏层节点数的列表
        :param activation: 激活函数
        """
        super().__init__()
        self.input_dim = input_dim
        self.net = mlp([input_dim] + list(hidden_sizes) + [2], activation, output_activation=nn.Identity, dropout=0.2)

    def forward(self, input):
        return torch.squeeze(self.net(input), -1)  # Critical to ensure right shaped: 2 value
