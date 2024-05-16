import torch
import torch.nn as nn


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)


class QNet(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(QNet, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden * 2),
                                     nn.ReLU(),
                                     nn.Linear(hidden * 2, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, a_num))

    def forward(self, s):
        return self.feature(s)
