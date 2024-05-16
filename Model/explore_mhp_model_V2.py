import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import init
from Model.missile_attack_zone_model import MissileHitProb
from Model.mlp.mlp_model import mlp
from Model.embedding.missile_states_embedding import MissileStateEmbedding
from Model.embedding.missile_alerts_embedding_simple import AlertEmbeddingSimple


class PredictModelSimple(nn.Module):
    def __init__(self, missile_attack_dim, device):
        super(PredictModelSimple, self).__init__()

        self.missile_attack_mlp = MissileHitProb(missile_attack_dim,
                                                 [512, 512, 512, 512, 512], activation=nn.LeakyReLU).to(device)

        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        if device == torch.device('cpu'):
            self.missile_attack_mlp.load_state_dict(
                torch.load(".\\output\\MissileAttack\\missile_attack_cpu.pt"))
        else:
            self.missile_attack_mlp.load_state_dict(
                torch.load(".\\output\\MissileAttack\\missile_attack_gpu.pt"))

    def forward(self, missile_attack_tensors):
        """
        :param missile_attack_tensors: batch_size * data_dim
        :return: # (batch size, 1)
        """
        missile_prob_out = self.missile_attack_mlp(missile_attack_tensors)  # (batch size, 2)
        return missile_prob_out


class RNDModelSimple(nn.Module):
    def __init__(self, missile_attack_dim, device=torch.device('cpu')):
        super(RNDModelSimple, self).__init__()

        self.predictor = PredictModelSimple(missile_attack_dim, device)

        self.device = device

    def forward(self, missile_attack_tensors):
        """

        :param missile_attack_tensors: batch_size * data_dim
        :return: (batch size, 1), (batch size, 1)
        """
        missile_attack_tensors = missile_attack_tensors.to(self.device)

        predict_feature = self.predictor.forward(missile_attack_tensors)

        return predict_feature
