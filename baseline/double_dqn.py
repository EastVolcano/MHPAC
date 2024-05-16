import copy
import torch
import torch.nn.functional as F
from .network import QNet
from .replayBuffer import ReplayBuffer
import numpy as np
import json


class DoubleDQN:
    def __init__(self,
                 path,
                 s_dim=13,
                 a_num=8,
                 hidden=256,
                 gamma=0.95,
                 capacity=int(1e5),
                 batch_size=512,
                 start_learn=2048,
                 lr=1e-4,
                 epsilon_start=0.1,
                 greedy_increase=5e-5,
                 epsilon_max=0.95,
                 replace_target_iter=512):

        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_num = a_num
        self.hidden = hidden
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.start_learn = start_learn
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.greedy_increase = greedy_increase
        self.epsilon_max = epsilon_max
        self.replace_target_iter = replace_target_iter
        self.train_it = 0

        # Network
        self.Q = QNet(s_dim, hidden, a_num)
        self.Q_target = QNet(s_dim, hidden, a_num)
        self.opt = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # replay buffer, or memory
        self.memory = ReplayBuffer(s_dim, capacity, batch_size)

    def get_action(self, s):
        """动作选择"""
        if np.random.rand() <= self.epsilon:
            s = torch.tensor(s, dtype=torch.float)
            actions_value = self.Q(s)
            action = torch.argmax(actions_value)
            action = action.item()
        else:
            action = np.random.randint(0, self.a_num)
        return action

    def store_transition(self, s, a, s_, r, done):
        """存储记忆，若记忆库中大于阈值，则进行学习"""
        self.memory.store_transition(s, a, s_, r, done)
        if self.memory.counter >= self.start_learn:
            s, a, s_, r, done = self.memory.get_sample()
            self._learn(s, a, s_, r, done)

    def _learn(self, s, a, s_, r, done):
        """参数更新"""
        self.train_it += 1
        # calculate loss function
        index = torch.tensor(range(len(r)), dtype=torch.long)
        q = self.Q(s)[index, a]
        with torch.no_grad():
            q_target = self.Q_target(s_)
            q_ = self.Q(s_)
            a_ = torch.max(q_, dim=1).indices
            td_target = r + (1 - done) * self.gamma * q_target[index, a_]
        loss = F.mse_loss(q, td_target)
        # train the network
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # renew epsilon
        self.epsilon = min(self.epsilon + self.greedy_increase, self.epsilon_max)
        # hard update
        if self.train_it % self.replace_target_iter == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def store_net(self, prefix):
        torch.save(self.Q.state_dict(), self.path + '/'+prefix+'_Q_Net.pth')

    def load_net(self, prefix):
        self.Q.load_state_dict(torch.load(self.path + '\\'+prefix+'_Q_Net.pth'))
        self.Q_target = copy.deepcopy(self.Q)
