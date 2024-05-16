"""多层感知机神经网络模型"""

import torch.nn as nn


def mlp(sizes, activation, output_activation=nn.Identity, dropout=0.05):
    """
    :param dropout: dropout比例
    :param sizes: 列表，长度表示隐藏层层数，元素表示该层神经元数
    :param activation: 隐藏层激活函数
    :param output_activation: 输出层激活函数
    :return: nn.Sequential
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Dropout(p=dropout), act()]
        else:
            if act is nn.Softmax:
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act(dim=0)]
            else:
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
