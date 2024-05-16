import os.path as osp
import os
import torch
from torch.optim import Adam
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from OnPolicyBuffer import PPOBuffer
from Spinup.mpi_torch_utils import num_procs, proc_id, mpi_avg_grads, sync_params, mpi_avg
from Spinup.torch_distribute import average_gradients_torch, average_x_torch, statistics_scalar_torch, sync_params_torch
from Utils.tensor_util import count_vars
from Model.actor_critic_mlp import ActorTwoCriticAlone
from Utils.optim_schedule import ScheduledOptim


class PPOAgent(object):
    def __init__(self, index, obs_self_dim, obs_target_dim, missile_state_dim, missile_alert_dim,
                 act_dim, args, trainable=True, torch_dist=False, device=torch.device('cuda:0')):
        # ----- set using cpu mpi or torch distributed -----#
        self.torch_dist = torch_dist
        # ----- set device -----#
        self.device = device
        # ----- set save path -----#
        self.save_dir = args.model_dir + str(args.seed) + "\\trained_model\\"
        os.makedirs(self.save_dir, exist_ok=True)
        # ----- set property -----#
        self.trainable = trainable
        self.act_dim = act_dim
        self.obs_self_dim = obs_self_dim
        self.obs_target_dim = obs_target_dim
        self.missile_state_dim = missile_state_dim
        self.missile_alert_dim = missile_alert_dim
        self.rnn_type = args.rnn_type

        # ----- set hyper param -----#
        self.epoch_train_iters = args.epoch_train_iters
        self.target_kl = args.target_kl
        self.clip_ratio = args.clip_ratio
        self.gamma = args.gamma
        self.gamma_rnd = args.gamma_rnd
        self.value_loss_weight = args.value_loss_weight
        self.task_weight = args.task_weight
        self.rnd_weight = args.rnd_weight
        self.bttp_seq_len = args.seq_len
        self.rnn_layer_num = args.rnn_layer_num
        self.rnn_hidden_size = args.rnn_hidden_size

        # ----- set algorithm variables -----#
        self.kl = None
        # setup experience buffer
        self.local_steps_per_epoch = int(args.steps / dist.get_world_size()) if torch_dist else (
            int(args.steps / num_procs()))
        self.buf = PPOBuffer(obs_self_dim, obs_target_dim, missile_state_dim, missile_alert_dim,
                             self.act_dim, (self.rnn_layer_num, self.rnn_hidden_size), self.local_steps_per_epoch,
                             gamma_rnd=args.gamma_rnd, gamma_task=args.gamma, lam=args.lam,
                             rnn_type=args.rnn_type, sequence_length=args.seq_len, torch_dist=self.torch_dist)

        # ----- Create actor and critic model -----#
        self.actor_critic = ActorTwoCriticAlone(self.obs_self_dim, self.obs_target_dim, act_dim,
                                                missile_state_dim, missile_alert_dim[0],
                                                hidden_shape=(self.rnn_layer_num, self.rnn_hidden_size),
                                                hidden_sizes=(128, 128, 128, 128), activation=nn.LeakyReLU,
                                                feature_layer=args.rnn_type, device=self.device)
        if self.trainable:
            # Set up optimizers for policy and value function
            self.ac_optimizer = Adam(self.actor_critic.parameters(), lr=args.lr, betas=(0.9, 0.999))
            # self.ac2_optimizer_schedule = ScheduledOptim(self.ac2_optimizer, d_model=args.scheduled_opt_dmodel,
            #                                              n_warmup_steps=args.warmup_steps, torch_dist=False)

            # Sync params across processes
            if torch_dist:
                sync_params_torch(self.actor_critic)
            else:
                sync_params(self.actor_critic.cpu())
                self.actor_critic.to(self.device)

            # Count variables
            self.var_counts = tuple(count_vars(module) for module in [self.actor_critic.actor, self.actor_critic.v_task,
                                                                      self.actor_critic.v_rnd])
            p_id = dist.get_rank() if self.torch_dist else proc_id()
            if (index == 0) and (p_id == 0):
                if self.rnn_type is None:
                    print("Total Parameters: actor", sum([p.nelement() for p in self.actor_critic.actor.parameters()]),
                          "critic: ", sum([p.nelement() for p in self.actor_critic.v_task.parameters()]),
                          "rnd v: ", sum([p.nelement() for p in self.actor_critic.v_rnd.parameters()]))
                else:
                    print("Total Parameters: total", sum([p.nelement() for p in self.actor_critic.parameters()]),
                          "RNN: ", sum([p.nelement() for p in self.actor_critic.actor.feature_module.parameters()]),
                          "actor: ", sum([p.nelement() for p in self.actor_critic.actor.parameters()]),
                          "critic: ", sum([p.nelement() for p in self.actor_critic.v_task.parameters()]),
                          "rnd v: ", sum([p.nelement() for p in self.actor_critic.v_rnd.parameters()]))
        # 迭代次数
        self.update_count = 0

    def compute_loss(self, data):
        # 将数据转入device中
        data = {key: value.to(self.device) for key, value in data.items()}

        if self.rnn_type is None:
            obs_self, obs_target, act = data['obs_self'], data['obs_target'], data['act']
            missile_states, missile_alerts = data['mis_state'], data['mis_alert']
            adv_task, adv_rnd = data['adv_task'], data['adv_rnd']
            logp_old, logp_mask = data['logp'], data['logp_mask']
            ret_task, ret_rnd = data['ret_task'], data['ret_rnd']
            adv_mask = data['adv_mask']

            # 结合两种奖励对应的优势函数
            adv = (self.task_weight * adv_task + self.rnd_weight * adv_rnd) / (self.rnd_weight + self.task_weight)

            # Policy loss
            # 推理
            pi, _, v_e, v_i, _, rnn_hidden = self.actor_critic.forward(obs_self, obs_target,
                                                                       missile_states, missile_alerts,
                                                                       hidden=None, seq_len=1)
            logp = pi.log_prob(act).sum(axis=-1)
            entropy = pi.entropy()

            # mask logp and adv
            logp = logp[logp_mask]
            logp_old = logp_old[logp_mask]
            adv = adv[adv_mask]
            # clip
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = - (torch.min(ratio * adv, clip_adv)).mean()
            loss_ent = - entropy.mean()
            policy_loss = loss_pi + 5e-3 * loss_ent
            # policy_loss = loss_pi

            # Task Critic Loss
            v_task_loss = ((v_e - ret_task) ** 2).mean()
            # RND Critic Loss
            v_rnd_loss = ((v_i - ret_rnd) ** 2).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=entropy.mean().item(), cf=clipfrac)

        elif self.rnn_type == 'lstm' or self.rnn_type == 'gru':
            obs_self, obs_target, act = data['obs_self'], data['obs_target'], data['act']
            missile_states, missile_alerts = data['mis_state'], data['mis_alert']
            if self.rnn_type == 'lstm':
                hx, cx, loss_mask = data['hx'], data['cx'], data['loss_mask']
            else:
                hx, loss_mask = data['hx'], data['loss_mask']
            adv_task, adv_rnd = data['adv_task'], data['adv_rnd']
            logp_old, logp_mask = data['logp'], data['logp_mask']
            ret_task, ret_rnd = data['ret_task'], data['ret_rnd']
            adv_mask = data['adv_mask']

            # 结合两种奖励对应的优势函数
            adv = (self.task_weight * adv_task + self.rnd_weight * adv_rnd) / (self.rnd_weight + self.task_weight)

            # Policy loss
            # lstm 延截断轨迹做推理
            if self.rnn_type == 'lstm':
                pi, _, v_e, v_i, _, rnn_hidden = self.actor_critic.forward(obs_self, obs_target,
                                                                           missile_states, missile_alerts,
                                                                           hidden=(hx, cx), seq_len=self.bttp_seq_len)
            else:
                pi, _, v_e, v_i, _, rnn_hidden = self.actor_critic.forward(obs_self, obs_target,
                                                                           missile_states, missile_alerts,
                                                                           hidden=hx, seq_len=self.bttp_seq_len)
            logp = pi.log_prob(act.reshape(act.shape[0] * act.shape[1], act.shape[2])).sum(axis=-1)
            entropy = pi.entropy()

            # 计入loss mask
            loss_mask = loss_mask.reshape(loss_mask.shape[0] * loss_mask.shape[1])
            v_e = v_e[loss_mask]
            v_i = v_i[loss_mask]
            entropy = entropy[loss_mask]
            # mask logp and adv
            logp_mask = logp_mask.reshape(logp_mask.shape[0] * logp_mask.shape[1])
            logp_old = logp_old.reshape(logp_old.shape[0] * logp_old.shape[1])
            logp = logp[logp_mask]
            logp_old = logp_old[logp_mask]
            adv = adv[adv_mask]

            # clip
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = - (torch.min(ratio * adv, clip_adv)).mean()
            loss_ent = - entropy.mean()
            policy_loss = loss_pi + 5e-3 * loss_ent

            # Task Critic Loss
            v_task_loss = ((v_e - ret_task) ** 2).mean()
            # RND Critic Loss
            v_rnd_loss = ((v_i - ret_rnd) ** 2).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=entropy.mean().item(), cf=clipfrac)

        else:
            raise ValueError("RNN 网络类型错误")

        return policy_loss, v_task_loss, v_rnd_loss, pi_info

    def update(self):
        self.update_count += 1

        if self.trainable:
            data = self.buf.get()

            pi_l_old, v_task_l_old, v_rnd_l_old, pi_info_old = self.compute_loss(data)
            pi_l_old, v_task_l_old, v_rnd_l_old = pi_l_old.item(), v_task_l_old.item(), v_rnd_l_old.item()

            # Train policy and two critic with multiple steps of gradient descent
            for i in range(self.epoch_train_iters):
                self.ac_optimizer.zero_grad()
                # self.ac2_optimizer_schedule.zero_grad()
                pi_l, v_task_l, v_rnd_l, pi_info = self.compute_loss(data)
                loss_all = pi_l + v_task_l + v_rnd_l

                if self.torch_dist:
                    self.kl = average_x_torch(pi_info['kl'])
                    self.kl = self.kl.item()
                else:
                    self.kl = mpi_avg(pi_info['kl'])

                if self.kl > self.target_kl:
                    # print('Early stopping at step %d due to reaching max kl.' % i)
                    break

                loss_all.backward()
                if self.torch_dist:
                    average_gradients_torch(self.actor_critic)  # average grads across distribute processes
                else:
                    mpi_avg_grads(self.actor_critic.cpu())  # average grads across MPI processes
                    self.actor_critic.to(self.device)

                self.ac_optimizer.step()
                # self.ac2_optimizer_schedule.step_and_update_lr()

            max_kl_stop_iter = i

            # Log changes from update
            ent, cf = pi_info_old['ent'], pi_info['cf']
            pi_l_new = pi_l.item()
            v_task_l_new = v_task_l.item()
            v_rnd_l_new = v_rnd_l.item()

            return (max_kl_stop_iter, pi_l_new, pi_l_old, v_task_l_new, v_task_l_old, v_rnd_l_new, v_rnd_l_old,
                    ent, self.kl, cf)
        else:
            pass

    def save(self):
        if self.rnn_type is None:
            torch.save(self.actor_critic.actor.state_dict(), self.save_dir + "pi_net.pt")
            torch.save(self.actor_critic.v_task.state_dict(), self.save_dir + "v_task_net.pt")
            torch.save(self.actor_critic.v_rnd.state_dict(), self.save_dir + "v_rnd_net.pt")
            torch.save(self.actor_critic.state_dict(), self.save_dir + "ac_net.pt")
        else:
            torch.save(self.actor_critic.actor.feature_module.state_dict(), self.save_dir + "rnn_feature.pt")
            torch.save(self.actor_critic.actor.state_dict(), self.save_dir + "pi_net.pt")
            torch.save(self.actor_critic.v_task.state_dict(), self.save_dir + "v_task_net.pt")
            torch.save(self.actor_critic.v_rnd.state_dict(), self.save_dir + "v_rnd_net.pt")
            torch.save(self.actor_critic.state_dict(), self.save_dir + "ac_net.pt")
