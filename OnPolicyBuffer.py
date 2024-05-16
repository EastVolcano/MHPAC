"""PPO 的经验存储"""

import numpy as np
import torch
from Spinup.torch_distribute import statistics_scalar_torch, average_x_torch, average_gradients_torch
from Spinup.mpi_torch_utils import mpi_statistics_scalar
from Utils.tensor_util import combined_shape, discount_cumsum, discount_cumsum_tensor


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_self_dim, obs_target_dim, missile_state_dim, missile_alert_dim, act_dim, hidden_shape, size,
                 gamma_rnd=0.99, gamma_task=0.995, lam=0.98,
                 rnn_type=None, sequence_length=16, torch_dist=False):
        # use cpu mpi or torch distributed
        self.torch_dist = torch_dist

        self.obs_self_dim = obs_self_dim
        self.obs_target_dim = obs_target_dim
        self.missile_state_dim = missile_state_dim
        self.missile_alert_dim = missile_alert_dim  # (18, 4)
        self.act_dim = act_dim

        self.obs_self_buf = np.zeros(combined_shape(size, obs_self_dim), dtype=np.float32)
        self.obs_target_buf = np.zeros(combined_shape(size, obs_target_dim), dtype=np.float32)
        self.missile_state_buf = np.zeros(combined_shape(size, missile_state_dim), dtype=np.int32)
        self.missile_alert_buf = np.zeros(combined_shape(size, missile_alert_dim), dtype=np.int32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_rnd_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rew_rnd_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ret_rnd_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.val_rnd_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.fighter_survive_buf = np.zeros(size, dtype=bool)
        # 智能体done的那一时刻依然为TRUE，从而done时刻的adv和logp不会被mask
        self.gamma_task, self.gamma_rnd, self.lam = gamma_task, gamma_rnd, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.rnn_type = rnn_type
        self.rnn_hx = np.zeros(combined_shape(size, hidden_shape), dtype=np.float32)
        self.rnn_cx = np.zeros(combined_shape(size, hidden_shape), dtype=np.float32)

        # sequence 列表
        self.obs_self_seq = []
        self.obs_target_seq = []
        self.mis_state_seq = []
        self.mis_alert_seq = []
        self.act_seq = []
        self.hx_seq = []
        self.cx_seq = []
        self.logp_seq = []
        self.loss_mask_seq = []
        self.logp_mask_seq = []  # 用来mask因飞机坠毁但它的导弹依然在飞行而产生的logp和adv, 包含了loss mask

        # BTTP 切分的序列长度
        self.seq_len = sequence_length

    def store(self, obs_self, obs_target, mis_state, mis_alert, act, rew, val, rew_rnd, val_rnd,
              logp, rnn_hidden, survive_info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # print("debug : ", self.ptr)
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_self_buf[self.ptr] = obs_self
        self.obs_target_buf[self.ptr] = obs_target
        self.missile_state_buf[self.ptr] = mis_state
        self.missile_alert_buf[self.ptr] = mis_alert
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.rew_rnd_buf[self.ptr] = rew_rnd
        self.val_rnd_buf[self.ptr] = val_rnd
        self.logp_buf[self.ptr] = logp
        # 存储隐状态
        if self.rnn_type == 'lstm':
            self.rnn_hx[self.ptr] = rnn_hidden[0]
            self.rnn_cx[self.ptr] = rnn_hidden[1]
        elif self.rnn_type == 'gru':
            self.rnn_hx[self.ptr] = rnn_hidden
        else:
            pass
        self.fighter_survive_buf[self.ptr] = survive_info
        self.ptr += 1

    def task_episodic_path(self, last_val=0):
        """
        Call this at the end of an episode trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma_task * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma_task * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma_task)[:-1]

        # 将episodic的轨迹切分成sequence
        if self.seq_len >= len(self.obs_target_buf[path_slice]):
            self.obs_self_seq.append(self.obs_self_buf[path_slice])
            self.obs_target_seq.append(self.obs_target_buf[path_slice])

            self.mis_state_seq.append(self.missile_state_buf[path_slice])
            self.mis_alert_seq.append(self.missile_alert_buf[path_slice])

            self.act_seq.append(self.act_buf[path_slice])

            self.hx_seq.append(self.rnn_hx[self.path_start_idx])
            if self.rnn_type == 'lstm':
                self.cx_seq.append(self.rnn_cx[self.path_start_idx])
            # loss mask
            self.loss_mask_seq.append([True for _ in range(self.path_start_idx, self.ptr)])
            # logp mask
            self.logp_mask_seq.append([(True and survive) for survive in self.fighter_survive_buf[path_slice]])
            self.logp_seq.append(self.logp_buf[path_slice])
            #
        else:
            for seq_start in range(self.path_start_idx, self.ptr, self.seq_len):
                seq_end = seq_start + self.seq_len if (seq_start + self.seq_len) < self.ptr else self.ptr

                self.obs_self_seq.append(self.obs_self_buf[seq_start: seq_end])
                # append array(<=seq_len, obs_self_dim)
                self.obs_target_seq.append(self.obs_target_buf[seq_start: seq_end])

                self.mis_state_seq.append(self.missile_state_buf[seq_start: seq_end])
                self.mis_alert_seq.append(self.missile_alert_buf[seq_start: seq_end])

                self.act_seq.append(self.act_buf[seq_start: seq_end])

                self.hx_seq.append(self.rnn_hx[seq_start])
                if self.rnn_type == 'lstm':
                    self.cx_seq.append(self.rnn_cx[seq_start])  # (rnn_num_layer, rnn_hidden_dim)
                # loss mask
                self.loss_mask_seq.append([True for _ in range(len(self.obs_target_seq[-1]))])
                # logp mask
                self.logp_mask_seq.append([(True and survive) for survive in
                                           self.fighter_survive_buf[seq_start: seq_end]])
                self.logp_seq.append(self.logp_buf[seq_start: seq_end])

        self.path_start_idx = self.ptr

    def rnd_non_episodic_path(self):
        """
        call this at the end of one epoch. This uses rnd rewards and rnd value estimates from
        the whole buffer to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        """
        rews = np.append(self.rew_rnd_buf, 0)
        vals = np.append(self.val_rnd_buf, 0)

        # adv
        deltas = rews[:-1] + self.gamma_rnd * vals[1:] - vals[:-1]
        self.adv_rnd_buf[:] = discount_cumsum(deltas, self.gamma_rnd * self.lam)

        # rewards-to-go
        self.ret_rnd_buf[:] = discount_cumsum(rews, self.gamma_rnd)[:-1]

    def get(self):
        """
        Call this at the end of an epoch to get all the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # 将存储的sequence填充到固定长度
        for o, seq_obs in enumerate(self.obs_target_seq):
            if len(seq_obs) < self.seq_len:
                self.obs_target_seq[o] = np.append(seq_obs,
                                                   np.array([np.zeros(self.obs_target_dim, dtype=np.float32)
                                                             for _ in range(self.seq_len - len(seq_obs))]), axis=0)
        for o, seq_obs in enumerate(self.obs_self_seq):
            if len(seq_obs) < self.seq_len:
                self.obs_self_seq[o] = np.append(seq_obs,
                                                 np.array([np.zeros(self.obs_self_dim, dtype=np.float32)
                                                           for _ in range(self.seq_len - len(seq_obs))]), axis=0)

        for m, seq_ms in enumerate(self.mis_state_seq):
            if len(seq_ms) < self.seq_len:
                self.mis_state_seq[m] = np.append(seq_ms,
                                                  np.array([np.zeros(self.missile_state_dim, dtype=np.int32)
                                                            for _ in range(self.seq_len - len(seq_ms))]), axis=0)
        for m, seq_al in enumerate(self.mis_alert_seq):
            if len(seq_al) < self.seq_len:
                self.mis_alert_seq[m] = np.append(seq_al,
                                                  np.array([np.zeros(self.missile_alert_dim, dtype=np.int32)
                                                            for _ in range(self.seq_len - len(seq_al))]), axis=0)

        for a, seq_act in enumerate(self.act_seq):
            if len(seq_act) < self.seq_len:
                self.act_seq[a] = np.append(seq_act, np.array([np.zeros(self.act_dim, dtype=np.float32)
                                                               for _ in range(self.seq_len - len(seq_act))]), axis=0)

        for m, seq_mask in enumerate(self.loss_mask_seq):
            if len(seq_mask) < self.seq_len:
                self.loss_mask_seq[m] += [False for _ in range(self.seq_len - len(seq_mask))]

        for m, logp_mask in enumerate(self.logp_mask_seq):
            if len(logp_mask) < self.seq_len:
                self.logp_mask_seq[m] += [False for _ in range(self.seq_len - len(logp_mask))]

        for l, seq_logp in enumerate(self.logp_seq):
            if len(seq_logp) < self.seq_len:
                self.logp_seq[l] = np.append(seq_logp,
                                             np.array([0. for _ in range(self.seq_len - len(seq_logp))]), axis=0)

        # Reshape the sequence to tensor
        obs_self_seq = torch.as_tensor(np.array(self.obs_self_seq), dtype=torch.float32)
        # size: (seq_num, seq_len, obs_self_dim)
        obs_target_seq = torch.as_tensor(np.array(self.obs_target_seq), dtype=torch.float32)
        # size: (seq_num, seq_len, obs_target_dim)

        mis_state_seq = torch.as_tensor(np.array(self.mis_state_seq), dtype=torch.int32)  # size: (seq_num, seq_len, 8)
        mis_alert_seq = torch.as_tensor(np.array(self.mis_alert_seq), dtype=torch.int32)  # size: (seq_num, seq_len, 8)

        act_seq = torch.as_tensor(np.array(self.act_seq), dtype=torch.float32)  # size: (seq_num, seq_len, act_dim)
        hx_seq = torch.as_tensor(np.array(self.hx_seq), dtype=torch.float32)  # size: (seq_num, num_layer, hidden_dim)
        if self.rnn_type == 'lstm':
            cx_seq = torch.as_tensor(np.array(self.cx_seq), dtype=torch.float32)
            # size: (seq_num, num_layer, hidden_dim)
        loss_mask_seq = torch.as_tensor(np.array(self.loss_mask_seq), dtype=torch.bool)  # size: (seq_num. seq_len)
        logp_mask_seq = torch.as_tensor(np.array(self.logp_mask_seq), dtype=torch.bool)  # size: (seq_num. seq_len)
        logp_seq = torch.as_tensor(np.array(self.logp_seq), dtype=torch.float32)  # size: (seq_num. seq_len)

        # Reshape buffer to tensor
        obs_self_buf = torch.as_tensor(self.obs_self_buf, dtype=torch.float32)
        obs_target_buf = torch.as_tensor(self.obs_target_buf, dtype=torch.float32)
        missile_state_buf = torch.as_tensor(self.missile_state_buf, dtype=torch.int32)
        missile_alert_buf = torch.as_tensor(self.missile_alert_buf, dtype=torch.int32)
        act_buf = torch.as_tensor(self.act_buf, dtype=torch.float32)
        ret_buf = torch.as_tensor(self.ret_buf, dtype=torch.float32)
        ret_rnd_buf = torch.as_tensor(self.ret_rnd_buf, dtype=torch.float32)
        adv_buf = torch.as_tensor(self.adv_buf, dtype=torch.float32)
        adv_rnd_buf = torch.as_tensor(self.adv_rnd_buf, dtype=torch.float32)
        logp_buf = torch.as_tensor(self.logp_buf, dtype=torch.float32)

        survive_buf = torch.as_tensor(self.fighter_survive_buf, dtype=torch.bool)

        # implement the advantage normalization trick
        if self.torch_dist:
            adv_mean, adv_std = statistics_scalar_torch(adv_buf)
            adv_buf = (adv_buf - adv_mean) / adv_std

            adv_rnd_mean, adv_rnd_std = statistics_scalar_torch(adv_rnd_buf)
            adv_rnd_buf = (adv_rnd_buf - adv_rnd_mean) / adv_rnd_std
        else:
            adv_mean, adv_std = mpi_statistics_scalar(adv_buf)
            adv_buf = (adv_buf - adv_mean) / adv_std

            adv_rnd_mean, adv_rnd_std = mpi_statistics_scalar(adv_rnd_buf)
            adv_rnd_buf = (adv_rnd_buf - adv_rnd_mean) / adv_rnd_std

        if self.rnn_type == 'lstm':
            data = dict(obs_self=obs_self_seq, obs_target=obs_target_seq, mis_state=mis_state_seq,
                        mis_alert=mis_alert_seq, act=act_seq, hx=hx_seq, cx=cx_seq,
                        ret_task=ret_buf, ret_rnd=ret_rnd_buf, adv_task=adv_buf, adv_rnd=adv_rnd_buf, logp=logp_seq,
                        loss_mask=loss_mask_seq, logp_mask=logp_mask_seq, adv_mask=survive_buf)

        elif self.rnn_type == 'gru':
            data = dict(obs_self=obs_self_seq, obs_target=obs_target_seq, mis_state=mis_state_seq,
                        mis_alert=mis_alert_seq, act=act_seq, hx=hx_seq,
                        ret_task=ret_buf, ret_rnd=ret_rnd_buf, adv_task=adv_buf, adv_rnd=adv_rnd_buf, logp=logp_seq,
                        loss_mask=loss_mask_seq, logp_mask=logp_mask_seq, adv_mask=survive_buf)
        else:
            data = dict(obs_self=obs_self_buf, obs_target=obs_target_buf, mis_state=missile_state_buf,
                        mis_alert=missile_alert_buf, act=act_buf, ret_task=ret_buf, ret_rnd=ret_rnd_buf,
                        logp_mask=survive_buf, adv_task=adv_buf, adv_rnd=adv_rnd_buf,
                        logp=logp_buf, adv_mask=survive_buf)
        # if torch.cuda.is_available():
        #     return {k: torch.as_tensor(v, dtype=torch.float32).cuda(device) for k, v in data.items()}
        # else:
        # for k, v in data.items():
        #     if (k == 'mis_state') or (k == 'mis_alert'):
        #         data[k] = torch.as_tensor(v, dtype=torch.int32)
        #     elif (k == 'logp_mask') or (k == 'loss_mask') or (k == 'adv_mask'):
        #         data[k] = torch.as_tensor(v, dtype=torch.bool)
        #     else:
        #         data[k] = torch.as_tensor(v, dtype=torch.float32)

        # 清空 sequence 列表
        self.obs_self_seq = []
        self.obs_target_seq = []
        self.mis_state_seq = []
        self.mis_alert_seq = []
        self.act_seq = []
        self.hx_seq = []
        self.cx_seq = []
        self.logp_seq = []
        self.loss_mask_seq = []
        self.logp_mask_seq = []  # 用来mask因飞机坠毁但它的导弹依然在飞行而产生的logp和adv, 包含了loss mask

        return data
