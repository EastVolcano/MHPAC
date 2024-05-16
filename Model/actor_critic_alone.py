"""基于nn.module的Actor神经网络模型"""

import torch
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
from Model.mlp.mlp_model import mlp
from gym.spaces import Box
from Model.embedding.missile_alerts_embedding_simple import AlertEmbeddingSimple
from Model.embedding.missile_states_embedding import MissileStateEmbedding


class Actor(nn.Module):
    """
    Actor基类
    """

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class RNNFeature(nn.Module):
    def __init__(self, input_dim, hidden_shape, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layer = hidden_shape[0]
        self.hidden_dim = hidden_shape[1]
        # self.feature_fc = nn.Linear(input_dim, 128)
        # self.feature_relu = nn.LeakyReLU()
        if rnn_type == 'lstm':
            self.feature_rnn = nn.LSTM(input_dim, hidden_shape[1], num_layers=hidden_shape[0], batch_first=True)
        elif rnn_type == 'gru':
            self.feature_rnn = nn.GRU(input_dim, hidden_shape[1], num_layers=hidden_shape[0], batch_first=True)
        else:
            raise ValueError("错误使用RNN特征提取层，或RNN网络类型错误")

    def forward(self, features_in, hidden, seq_len, batch=True):
        """

        :param seq_len: 输入的序列长度
        :param features_in: batch为True: (seq_num * seq_len, input_dim); batch为 False: (input_dim, )
        :param hidden: lstm: (hx, cx); gru: hx; batch为 False: size=(num_layer, hidden_dim);
        batch为True：size=(seq_num, num_layer, hidden_dim)
        :param batch: True (网络更新)； False (与环境交互)
        :return:
        """
        if (not batch) and seq_len > 1:
            raise ValueError('不是网络更新时，序列长度应为1')
        # # 线性层
        # features = self.feature_relu(self.feature_fc(features_in))
        # RNN
        if not batch:
            # 与环境交互时，rnn的输入需要扩展到（seq_len=1, data）
            if self.rnn_type == 'lstm' or self.rnn_type == 'gru':
                features, rnn_hidden = self.feature_rnn(features_in.unsqueeze(0), hidden)
            else:
                features, rnn_hidden = None, None
        else:
            # 训练时：隐状态需要reshape为(num_layer, seq_num, hidden_dim)
            if self.rnn_type == 'lstm':
                features, rnn_hidden = self.feature_rnn(features_in.reshape(features_in.shape[0] // seq_len, seq_len,
                                                                            features_in.shape[1]),
                                                        (hidden[0].reshape(self.num_layer, hidden[0].shape[0],
                                                                           self.hidden_dim),
                                                         hidden[1].reshape(self.num_layer, hidden[1].shape[0],
                                                                           self.hidden_dim)))
            elif self.rnn_type == 'gru':
                features, rnn_hidden = self.feature_rnn(features_in.reshape(features_in.shape[0] // seq_len, seq_len,
                                                                            features_in.shape[1]),
                                                        hidden.reshape(self.num_layer, hidden.shape[0],
                                                                       self.hidden_dim))
            else:
                features, rnn_hidden = None, None
        return features, rnn_hidden


class MLPGaussianActor(Actor):
    """
    连续动作空间的多层感知机Actor网络模型
    """

    def __init__(self, feature_layer, alert_num_embedding, obs_dim_self, obs_dim_target, act_dim,
                 state_embed_size, alert_embed_size, hidden_sizes, rnn_hidden_shape, activation, device):
        super().__init__()
        self.device = device
        # attribute
        self.feature_layer = feature_layer
        self.rnn_num_layer = rnn_hidden_shape[0]
        self.rnn_hidden_dim = rnn_hidden_shape[1]

        # missile embedding
        self.alert_embed = AlertEmbeddingSimple(num_embeddings=alert_num_embedding,
                                                embed_size=alert_embed_size).to(device)
        self.alert_flatten = nn.Flatten(1, -1).to(device)  # out: (batch_size, 2*alert_embed_size)

        self.missile_state_embed = MissileStateEmbedding(embed_size=state_embed_size).to(device)
        self.mis_state_flatten = nn.Flatten(1, -1).to(device)  # out: (batch_size, 4*state_embed_size)

        if feature_layer is None:
            self.mu_net = mlp([obs_dim_self + obs_dim_target + 2 * alert_embed_size + 4 * state_embed_size] +
                              list(hidden_sizes) + [act_dim],
                              activation, output_activation=nn.Tanh)

        elif feature_layer == 'lstm' or feature_layer == 'gru':
            # 特征提取层
            # 导弹状态也作为序列输入RNN，使agent能记忆相对当前时刻，什么时候发射了导弹、什么时候发射的导弹命中或失效
            self.feature_module = RNNFeature(input_dim=obs_dim_self + obs_dim_target +
                                                       2 * alert_embed_size + 4 * state_embed_size,
                                             hidden_shape=rnn_hidden_shape,
                                             rnn_type=feature_layer).to(device)
            self.mu_net = mlp([rnn_hidden_shape[1]] + list(hidden_sizes) + [act_dim],
                              activation, output_activation=nn.Tanh)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def get_forward(self, obs_self, obs_target, missile_states, missile_alerts, hidden=None, seq_len=1):

        output, rnn_hidden = None, None

        if self.feature_layer is None:
            if len(missile_alerts.size()) < 3:
                mis_state = self.missile_state_embed(missile_states.unsqueeze(0))
                mis_alert = self.alert_embed(missile_alerts.unsqueeze(0))
            else:
                mis_state = self.missile_state_embed(missile_states)  # (batch_size, 4, state_embed_size) /
                mis_alert = self.alert_embed(missile_alerts)  # (batch_size, 4, 2, alert_embed_size)

            mis_state = self.mis_state_flatten(mis_state)  # (batch_size, 4*state_embed_size)
            mis_alert = mis_alert.sum(dim=-3)  # (batch_size, 2, alert_embed_size)
            mis_alert = self.alert_flatten(mis_alert)  # (batch_size, 2*alert_embed_size)

            mis_state, mis_alert, obs_self, obs_target = (mis_state.squeeze(), mis_alert.squeeze(),
                                                          obs_self.squeeze(), obs_target.squeeze())
            output = self.mu_net(torch.concat((obs_self, mis_state, obs_target, mis_alert), dim=-1))

        elif (self.feature_layer == 'lstm') or (self.feature_layer == 'gru'):
            # 与环境交互时的推理
            if seq_len == 1 and len(obs_target.size()) < 3:
                mis_state = self.missile_state_embed(missile_states.unsqueeze(0))  # (1, 4, state_embed_size)
                mis_alert = self.alert_embed(missile_alerts.unsqueeze(0))  # (1, 4, 2, alert_embed_size)

                mis_state = self.mis_state_flatten(mis_state)  # (1, 4*state_embed_size)
                mis_alert = mis_alert.sum(dim=-3)  # (1,2,alert_embed_size)
                mis_alert = self.alert_flatten(mis_alert)  # (1,2*alert_embed_size)

                mis_state, mis_alert = mis_state.squeeze(), mis_alert.squeeze()
                # (4*state_embed_size, ), (2*alert_embed_size, )

                features_in = torch.concat((obs_self, mis_state, obs_target, mis_alert), dim=-1)
                # obs_dim_self + obs_dim_target + 2 * alert_embed_size + 4 * state_embed_size
                features, rnn_hidden = self.feature_module.forward(features_in, hidden, seq_len=1, batch=False)
                # features: (seq_len(1), hidden_size), rnn_hidden: (num_layer, hidden_size)

                output = self.mu_net(features.squeeze())

            else:  # 训练时的推理
                mis_state = self.missile_state_embed(missile_states.reshape(
                    missile_states.shape[0] * missile_states.shape[1],
                    missile_states.shape[2]))  # (seq_num * seq_len, 4, state_embed_size)
                mis_alert = self.alert_embed(missile_alerts.reshape(
                    missile_alerts.shape[0] * missile_alerts.shape[1],
                    missile_alerts.shape[2], missile_alerts.shape[3]))  # (seq_num * seq_len, 4, 2, alert_embed_size)

                mis_state = self.mis_state_flatten(mis_state)  # (seq_num * seq_len, 4*state_embed_size)
                mis_alert = mis_alert.sum(dim=-3)  # (seq_num * seq_len, 2, alert_embed_size)
                mis_alert = self.alert_flatten(mis_alert)  # (seq_num * seq_len, 2*alert_embed_size)

                obs_self_in = obs_self.reshape(obs_self.shape[0] * obs_self.shape[1], obs_self.shape[2])
                obs_target_in = obs_target.reshape(obs_target.shape[0] * obs_target.shape[1], obs_target.shape[2])
                # (seq_num * seq_len, obs_dim_target)
                features_in = torch.concat((obs_self_in, mis_state, obs_target_in, mis_alert), dim=-1)
                # (seq_num * seq_len, obs_dim_self + obs_dim_target + 2 * alert_embed_size + 4 * state_embed_size)
                features, rnn_hidden = self.feature_module.forward(features_in, hidden, seq_len=seq_len, batch=True)
                # features: (seq_num, seq_len, hidden_size), rnn_hidden: (num_layer, seq_num, hidden_size)

                output = self.mu_net(features.reshape(features.shape[0] * features.shape[1], features.shape[2]))

        return output, rnn_hidden

    def _distribution(self, forward_out):
        mu = forward_out
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    """
    多层感知机Critic网络模型
    """

    def __init__(self, feature_layer, alert_num_embedding, obs_dim_self, obs_dim_target,
                 state_embed_size, alert_embed_size,
                 hidden_sizes, activation, device):
        super().__init__()
        self.feature_layer = feature_layer

        # missile embedding
        self.alert_embed = AlertEmbeddingSimple(num_embeddings=alert_num_embedding,
                                                embed_size=alert_embed_size).to(device)
        self.alert_flatten = nn.Flatten(1, -1).to(device)  # out: (batch_size, 2*alert_embed_size)

        self.missile_state_embed = MissileStateEmbedding(embed_size=state_embed_size).to(device)
        self.mis_state_flatten = nn.Flatten(1, -1).to(device)  # out: (batch_size, 4*state_embed_size)

        self.net = mlp([obs_dim_self + obs_dim_target + 2 * alert_embed_size + 4 * state_embed_size]
                       + list(hidden_sizes) + [1], activation)

    def forward(self, obs_self, obs_target, missile_states, missile_alerts, seq_len):
        v = None

        if self.feature_layer is None:

            if len(missile_alerts.size()) < 3:
                mis_state = self.missile_state_embed(missile_states.unsqueeze(0))
                mis_alert = self.alert_embed(missile_alerts.unsqueeze(0))
            else:
                mis_state = self.missile_state_embed(missile_states)  # (batch_size, 4, state_embed_size) /
                mis_alert = self.alert_embed(missile_alerts)  # (batch_size, 4, 2, alert_embed_size)

            mis_state = self.mis_state_flatten(mis_state)  # (batch_size, 4*state_embed_size)
            mis_alert = mis_alert.sum(dim=-3)  # (batch_size, 2, alert_embed_size)
            mis_alert = self.alert_flatten(mis_alert)  # (batch_size, 2*alert_embed_size)

            mis_state, mis_alert, obs_self, obs_target = (mis_state.squeeze(), mis_alert.squeeze(),
                                                          obs_self.squeeze(), obs_target.squeeze())
            v = self.net(torch.concat((obs_self, mis_state, obs_target, mis_alert), dim=-1))

        elif (self.feature_layer == 'lstm') or (self.feature_layer == 'gru'):

            # 与环境交互时的推理
            if seq_len == 1 and len(obs_target.size()) < 3:
                mis_state = self.missile_state_embed(missile_states.unsqueeze(0))  # (1, 4, state_embed_size)
                mis_alert = self.alert_embed(missile_alerts.unsqueeze(0))  # (1, 4, 2, alert_embed_size)

                mis_state = self.mis_state_flatten(mis_state)  # (1, 4*state_embed_size)
                mis_alert = mis_alert.sum(dim=-3)  # (1,2,alert_embed_size)
                mis_alert = self.alert_flatten(mis_alert)  # (1,2*alert_embed_size)

                mis_state, mis_alert = mis_state.squeeze(), mis_alert.squeeze()
                # (4*state_embed_size, ), (2*alert_embed_size, )

                v = self.net(torch.concat((obs_self, mis_state, obs_target, mis_alert), dim=-1))

            else:  # 训练时的推理
                mis_state = self.missile_state_embed(missile_states.reshape(
                    missile_states.shape[0] * missile_states.shape[1],
                    missile_states.shape[2]))  # (seq_num * seq_len, 4, state_embed_size)
                mis_alert = self.alert_embed(missile_alerts.reshape(
                    missile_alerts.shape[0] * missile_alerts.shape[1],
                    missile_alerts.shape[2], missile_alerts.shape[3]))  # (seq_num * seq_len, 4, 2, alert_embed_size)

                mis_state = self.mis_state_flatten(mis_state)  # (seq_num * seq_len, 4*state_embed_size)
                mis_alert = mis_alert.sum(dim=-3)  # (seq_num * seq_len, 2, alert_embed_size)
                mis_alert = self.alert_flatten(mis_alert)  # (seq_num * seq_len, 2*alert_embed_size)

                obs_self_in = obs_self.reshape(obs_self.shape[0] * obs_self.shape[1], obs_self.shape[2])
                # (seq_num * seq_len, obs_dim_self)
                obs_target_in = obs_target.reshape(obs_target.shape[0] * obs_target.shape[1], obs_target.shape[2])
                # (seq_num * seq_len, obs_dim_target)

                v = self.net(torch.concat((obs_self_in, mis_state, obs_target_in, mis_alert), dim=-1))

        return torch.squeeze(v, -1)  # Critical to ensure v has right shape.


class ActorTwoCriticAlone(nn.Module):
    """
    1个actor(连续动作空间)，2个critic的网络模型
    obs_dim_self: 对自身飞行状态的观测
    obs_dim_target：对目标的观测
    feature_layer: None(不使用), 'lstm', 'gru'
    """

    def __init__(self, obs_dim_self, obs_dim_target, act_dim, state_embed_size, alert_embed_size, hidden_shape=(1, 128),
                 hidden_sizes=(64, 64), activation=nn.LeakyReLU, feature_layer=None, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.rnn_num_layer = hidden_shape[0]
        self.rnn_hidden_dim = hidden_shape[1]

        self.feature_layer = feature_layer
        if feature_layer is None:
            # actor model
            self.actor = MLPGaussianActor(feature_layer, 101, obs_dim_self, obs_dim_target, act_dim,
                                          state_embed_size, alert_embed_size, hidden_sizes, hidden_shape, activation,
                                          device).to(device)

        elif feature_layer == 'lstm' or feature_layer == 'gru':
            # Actor有特征提取层
            # 导弹状态也作为序列输入RNN，使agent能记忆相对当前时刻，什么时候发射了导弹、什么时候发射的导弹命中或失效
            self.actor = MLPGaussianActor(feature_layer, 101, obs_dim_self, obs_dim_target, act_dim,
                                          state_embed_size, alert_embed_size, hidden_sizes, hidden_shape, activation,
                                          device).to(device)

        else:
            raise ValueError('没有定义特征提取层的类型, 或定义错误')

        # build task critic model
        self.v_task = MLPCritic(feature_layer, 101, obs_dim_self, obs_dim_target,
                                state_embed_size, alert_embed_size,
                                hidden_sizes, activation, device).to(device)

        # build value function for network distillation
        self.v_rnd = MLPCritic(feature_layer, 101, obs_dim_self, obs_dim_target,
                               state_embed_size, alert_embed_size,
                               hidden_sizes, activation, device).to(device)

    def forward(self, obs_self, obs_target, missile_states, missile_alerts, hidden=None, seq_len=1):
        """

        :param missile_alerts: 与环境交互 size: (8, ); 训练 size: (batch_size, 8) / (seq_num, seq_len, 8)
        :param missile_states: 与环境交互 size: (4, 2); 训练 size: (batch_size, 4, 2) / (seq_num, seq_len, 4, 2)
        :param obs_self: 与环境交互：seq_len=1, size=(obs_self_dim,);
        训练：seq_len>1, size: (seq_num, seq_len, obs_self_dim) / (batch_size, obs_self_dim)
        :param obs_target: 与环境交互：seq_len=1, size=(obs_target_dim,);
        训练：seq_len>1, size: (seq_num, seq_len, obs_target_dim) / (batch_size, obs_target_dim)
        :param hidden: 与环境交互: size=(num_layer, hidden_dim); 训练： size=(seq_num, num_layer, hidden_dim)
        :param seq_len: batch_size = seq_len * seq_num - len(loss_mask==False)
        :return:
        """
        actor_net_out, actor_rnn_hidden = self.actor.get_forward(obs_self, obs_target, missile_states, missile_alerts,
                                                                 hidden, seq_len)
        pi = self.actor._distribution(actor_net_out)
        a = pi.sample()
        logp_a = self.actor._log_prob_from_distribution(pi, a)

        v_e = self.v_task(obs_self, obs_target, missile_states, missile_alerts, seq_len)
        v_i = self.v_rnd(obs_self, obs_target, missile_states, missile_alerts, seq_len)

        return pi, a, v_e, v_i, logp_a, actor_rnn_hidden
