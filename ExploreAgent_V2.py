import os, csv
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from Model.explore_mhp_model_V2 import RNDModelSimple
from Utils.tensor_util import combined_shape
from Observation.featured_observation_V2 import logistic
from Spinup.mpi_torch_utils import num_procs, proc_id, mpi_avg_grads, sync_params, mpi_avg
from Spinup.torch_distribute import average_gradients_torch, average_x_torch, statistics_scalar_torch, sync_params_torch


class RNDBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, missile_attack_dim, size):
        self.missile_attack_dim = missile_attack_dim

        self.missile_attack_buf = np.zeros(combined_shape(size, missile_attack_dim), dtype=np.float32)
        self.missile_label_buf = np.zeros(combined_shape(size, 2), dtype=np.float32)

        self.data_buff = []
        print("start loading missile attack data: ")
        name = os.listdir(".\\DataSet\\missile_data\\")
        for i, b in enumerate(name):
            with open(".\\DataSet\\missile_data\\" + name[i], "r", newline='') as csvfile:
                reader = csv.reader(csvfile)
                for j, row in enumerate(reader):
                    if j > 1:
                        self.data_buff.append(row)
            csvfile.close()

        self.ptr, self.max_size = 0, size

    def store(self, missile_attack_input, missile_attack_label):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # print("debug : ", self.ptr)
        assert self.ptr <= self.max_size  # buffer has to have room so you can store

        if self.ptr == self.max_size:
            pass
        else:
            self.missile_attack_buf[self.ptr] = missile_attack_input
            self.missile_label_buf[self.ptr] = missile_attack_label
            self.ptr += 1

    def get(self):
        """
        Call this at the end of an epoch to get all the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr <= self.max_size  # buffer has to be full or smaller before you can get

        if self.ptr < self.max_size:
            # 收集的样本不够，则从原训练集中随机抽取数据补满，帮助网络复习原有知识
            data_review_num = self.max_size - self.ptr
            for dr in range(data_review_num):
                attack_zone_input = []
                attack_zone_label = []
                data_array = self.data_buff[np.random.randint(0, len(self.data_buff), size=1)[0]]
                for i, data in enumerate(data_array):
                    if i == 0:  # alt
                        attack_zone_input.append(0.000105 * float(data) if float(data) < 2000
                                                 else logistic(float(data), 0.00053, 4500))
                    elif i == 1:  # fire_ma
                        attack_zone_input.append(logistic(float(data), 7, 0.75))
                    elif i == 2:  # fire_pitch
                        attack_zone_input.append(float(data) / 90)
                    elif i == 3:  # dist
                        attack_zone_input.append(2 * logistic(float(data), 0.00076, 0) - 1)
                    elif i == 4:  # body_q_t
                        attack_zone_input.append(float(data) / 180)
                    elif i == 5:  # body_q_d
                        attack_zone_input.append(float(data) / 90)
                    elif i == 6:  # target_ma
                        attack_zone_input.append(logistic(float(data), 7, 0.75))
                    elif i == 7:  # target_hori_AA
                        attack_zone_input.append(float(data) / 180)
                    elif i == 8:  # target_v_pitch
                        attack_zone_input.append(float(data) / 90)
                    elif i == 9:  # hit
                        if data == 'False':
                            attack_zone_label = [0, 1]
                        elif data == 'True':
                            attack_zone_label = [1, 0]
                self.missile_attack_buf[self.ptr + dr] = attack_zone_input
                self.missile_label_buf[self.ptr + dr] = attack_zone_label

        data = dict(missile_attack=self.missile_attack_buf, missile_label=self.missile_label_buf)

        # reset ptr
        self.ptr = 0
        # reshape data to tensor
        for k, v in data.items():
            if k == 'missile_attack':
                data[k] = torch.as_tensor(v, dtype=torch.float32)
            elif k == 'missile_label':
                data[k] = torch.as_tensor(v, dtype=torch.float32)
        return data


class RND(object):
    def __init__(self, index, missile_attack_dim, args, trainable, torch_dist=False, device=torch.device('cuda:0')):
        self.index = index
        self.torch_dist = torch_dist
        self.missile_attack_dim = missile_attack_dim
        self.trainable = trainable
        self.device = device
        self.rnd_model = RNDModelSimple(missile_attack_dim, self.device)
        self.save_dir = args.model_dir + str(args.seed) + "\\trained_model\\"

        # 用来记录一局对战中，导弹发射时的状态和命中情况
        self.fire_record = {"mis_fire_state": np.zeros((4, missile_attack_dim)), "mis_state": np.zeros(4)}

        # 同步不同进程的模型参数
        if self.torch_dist:
            sync_params_torch(self.rnd_model)
        else:
            sync_params(self.rnd_model.cpu())
            self.rnd_model.to(self.device)

        buf_size = 256
        self.train_data_buffer = RNDBuffer(missile_attack_dim, buf_size)

        self.optimizer = Adam(self.rnd_model.predictor.parameters(), lr=args.predictor_lr, betas=(0.9, 0.999))
        self.loss_func = nn.CrossEntropyLoss()

    def compute_cross_entropy_error(self, missile_attack_tensor, missile_label_tensor):
        # print(f"debug device: mis_attack : {missile_attack_tensors.device}, other: {other_continuous_tensors.device}, "
        #       f"mis_state: {missile_state_tensors.device}, missile_alert_tensors: {missile_alert_tensors.device}")
        # 设置模型为eval模式
        self.rnd_model.eval()
        # 用于推理，计算RND奖励
        predict = self.rnd_model.forward(missile_attack_tensor)
        ce = torch.softmax(predict, dim=-1)
        cross_entropy_error = - torch.sum(torch.mul(torch.log(torch.softmax(predict, dim=-1)),
                                                    missile_label_tensor), dim=-1)
        cross_entropy_error = cross_entropy_error.item()
        return cross_entropy_error

    def compute_cross_entropy_loss(self, missile_attack_tensors, missile_label_tensors):
        # print(f"debug device: mis_attack : {missile_attack_tensors.device}, other: {other_continuous_tensors.device}, "
        #       f"mis_state: {missile_state_tensors.device}, missile_alert_tensors: {missile_alert_tensors.device}")

        # 训练预测网络，计算loss
        predict = self.rnd_model.forward(missile_attack_tensors)
        cross_entropy_loss = self.loss_func(predict, missile_label_tensors)
        return cross_entropy_loss

    def update_predictor(self):
        # 设置为train模式
        self.rnd_model.train()
        # 获取数据
        data = self.train_data_buffer.get()
        data = {key: value.to(self.device) for key, value in data.items()}
        missile_attack_tensors, missile_label_tensors = data['missile_attack'], data['missile_label']

        self.optimizer.zero_grad()
        # 计算loss, 反向传播
        loss = self.compute_cross_entropy_loss(missile_attack_tensors, missile_label_tensors)
        loss.backward()
        # 平均各进程梯度
        if self.torch_dist:
            average_gradients_torch(self.rnd_model.predictor)
        else:
            mpi_avg_grads(self.rnd_model.predictor.cpu())
            self.rnd_model.to(self.device)
        self.optimizer.step()

    def save(self):
        torch.save(self.rnd_model.state_dict(), self.save_dir + "missile_attack_" + str(self.index) + ".pt")
