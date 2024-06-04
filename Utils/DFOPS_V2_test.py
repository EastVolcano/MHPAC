import numpy as np
import random
import torch
import torch.nn as nn
from WVRENV_PHD.utils.GNCData import wgs84ToNED, euler2vector
from Model.actor_critic_alone import ActorTwoCriticAlone
from Utils.SECA import Engagement_wvr
from baseline.warcraft_red import Warcraft, wgs84_2_ned, angle_error
from baseline.double_dqn import DoubleDQN
from baseline.DQN_Env import get_state
from Utils.tensor_util import count_vars


def boltzmann(value_list, temp):
    psum = np.sum(np.exp(value_list / (temp + 0.001)))
    p_dist = []
    for a in value_list:
        p_dist.append((np.exp(a / (temp + 0.001))) / (psum + 0.001))
    return p_dist


class DFOPS(object):

    def __init__(self, self_play_dir, rnn_type, rnn_hidden_shape, init_kar, proc_id=0,
                 net_device=torch.device('cuda:0')):

        self.time_count = 0
        self.init_kar = init_kar
        self.rnn_type = rnn_type
        self.rnn_hidden_shape = rnn_hidden_shape
        self.self_play_dir = self_play_dir
        self.proc_id = proc_id
        self.net_device = net_device
        self.SECA = Engagement_wvr(index=1)
        self.state_machine_DQN = Warcraft()
        self.DQN = DoubleDQN("E:\\ReinforcementLearning\\MHPAC\\baseline\\Record0")
        self.DQN.load_net(prefix='comprehensive')
        self.DQN_a_id = 0
        # 对手策略集
        new_opponent_dict = {'guidance': False,
                             'Pram,net': ActorTwoCriticAlone(14, 8, 5, 8, 8,
                                                             hidden_shape=self.rnn_hidden_shape,
                                                             hidden_sizes=(128, 128, 128, 128),
                                                             activation=nn.LeakyReLU,
                                                             feature_layer='lstm',
                                                             device=self.net_device),
                             'index': 1}
        new_opponent_dict['Pram,net'].load_state_dict(torch.load("E:\\ReinforcementLearning\\MHPAC\\output\\baseline"
                                                                 + "\\r_ppo\\ActorCritic\\0\\trained_model"
                                                                 + "\\ac_net.pt"))

        new_opponent_dict_2 = {'guidance': False,
                               'Pram,net': ActorTwoCriticAlone(14, 8, 5, 8, 8,
                                                               hidden_shape=self.rnn_hidden_shape,
                                                               hidden_sizes=(128, 128, 128, 128),
                                                               activation=nn.LeakyReLU,
                                                               feature_layer='lstm',
                                                               device=self.net_device),
                               'index': 2}
        new_opponent_dict_2['Pram,net'].load_state_dict(torch.load("E:\\ReinforcementLearning\\MHPAC\\output\\"
                                                                   + "baseline\\lstm_mhpac\\trained_model"
                                                                   + "\\ac_net.pt"))
        new_opponent_dict_3 = {'guidance': False,
                               'Pram,net': ActorTwoCriticAlone(14, 8, 5, 8, 8,
                                                               hidden_shape=self.rnn_hidden_shape,
                                                               hidden_sizes=(128, 128, 128, 128),
                                                               activation=nn.LeakyReLU,
                                                               feature_layer=None,
                                                               device=self.net_device),
                               'index': 3}
        new_opponent_dict_3['Pram,net'].load_state_dict(torch.load("E:\\ReinforcementLearning\\MHPAC\\output\\"
                                                                   + "baseline\\wo_mhpac\\trained_model"
                                                                   + "\\ac_net.pt"))
        self.opponent_list = [{"guidance": True, "Pram,net": 0.0, "index": 0},
                              new_opponent_dict, new_opponent_dict_2, new_opponent_dict_3,
                              {"guidance": None, "Pram,net": 0.0, "index": 4},
                              {"guidance": 'HRL', "Pram,net": 0.0, "index": 5},
                              {"guidance": True, "Pram,net": 0.9, "index": 6}
                              ]
        # 对手策略得分
        self.sample_scores = np.array([1, 1, 1, 1, 1])
        self.opponent_id = 5
        # 自博弈历史策略编号
        self.self_play_index = 1  # 用于放到对手策略池中的历史策略编号
        # 目标视线
        self.target_los = [1, 0, 0]

        if rnn_type is None:
            self.rnn_hidden_state = None
        elif rnn_type == 'lstm':
            self.rnn_hidden_state = (torch.zeros(rnn_hidden_shape, dtype=torch.float32, device=self.net_device),
                                     torch.zeros(rnn_hidden_shape, dtype=torch.float32, device=self.net_device))
        elif rnn_type == 'gru':
            self.rnn_hidden_state = torch.zeros(rnn_hidden_shape, dtype=torch.float32, device=self.net_device)

    def meta_solver(self):
        """
        在环境env重置后调用
        :return:
        """
        self.time_count = 0
        self.DQN_a_id = 0
        self.state_machine_DQN = Warcraft()
        self.SECA = Engagement_wvr(index=1)
        self.opponent_id = 5

        if self.rnn_type is None:
            self.rnn_hidden_state = None
        elif self.rnn_type == 'lstm':
            self.rnn_hidden_state = (torch.zeros(self.rnn_hidden_shape, dtype=torch.float32, device=self.net_device),
                                     torch.zeros(self.rnn_hidden_shape, dtype=torch.float32, device=self.net_device))
        elif self.rnn_type == 'gru':
            self.rnn_hidden_state = torch.zeros(self.rnn_hidden_shape, dtype=torch.float32, device=self.net_device)

    def sampled_policy(self, fighter, opponent, maneuver_lib, fighter_obs, step_time, mis_model, fighter_rnd_obs):
        if not (opponent.index in fighter.sensors.eodas_list):
            if step_time % 50 == 0:
                l_n, l_e, l_d = wgs84ToNED(opponent.fc_data.fLatitude, opponent.fc_data.fLongitude,
                                           opponent.fc_data.fAltitude,
                                           fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                           fighter.fc_data.fAltitude)
                self.target_los = [l_n, l_e, l_d]
            else:
                pass
        else:
            l_n, l_e, l_d = wgs84ToNED(opponent.fc_data.fLatitude, opponent.fc_data.fLongitude,
                                       opponent.fc_data.fAltitude,
                                       fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                       fighter.fc_data.fAltitude)
            self.target_los = [l_n, l_e, l_d]

        if self.opponent_list[self.opponent_id]['guidance'] == 'HRL':
            if self.time_count % 10 == 0:
                dqn_s = get_state(fighter, opponent)
                self.DQN_a_id = self.DQN.get_action(dqn_s)
            fighter_action = self.state_machine_DQN.aircraft(fighter, opponent, given_mode=self.DQN_a_id+1)
            ctrl = fighter_action
            thrust = ctrl[2] / 100
            load = ctrl[0]
            omega = ctrl[1]
            rudder = 0

            mis_model.eval()
            mis_attack = torch.as_tensor(fighter_rnd_obs["mis_attack"], dtype=torch.float32, device=mis_model.device)
            with torch.no_grad():
                mis_prob = torch.softmax(mis_model.forward(mis_attack), dim=-1)[0]
                mis_prob = mis_prob.cpu().item()
            if (len(fighter_obs['mis_states'][np.where(fighter_obs['mis_states'] == 1)]) > 0) and (mis_prob < 0.9):
                fire_missile = 0
            elif (len(fighter_obs['mis_states'][np.where(fighter_obs['mis_states'] == 0)]) < 2) and (mis_prob < 0.9):
                fire_missile = 0
            else:
                fire_missile = int(mis_prob > 0.5)

        elif self.opponent_list[self.opponent_id]['guidance'] is None:
            ctrl = self.SECA.statemachine_wvr(fighter=fighter, enemy=opponent)
            thrust = ctrl[0]
            load = (ctrl[1] * 9) if ctrl[1] > 0 else (ctrl[1] * 3)
            omega = ctrl[2] * 300
            rudder = ctrl[3]
            fire_missile = self.SECA.missile_fire_close

        elif self.opponent_list[self.opponent_id]['guidance']:
            # 制导策略的对手
            thrust, load, omega, rudder = (
                maneuver_lib.kar_ppg(fighter=fighter,
                                     kar=self.opponent_list[self.opponent_id]['Pram,net'],
                                     k=2.8, target_los=self.target_los))
            mis_model.eval()
            mis_attack = torch.as_tensor(fighter_rnd_obs["mis_attack"], dtype=torch.float32, device=mis_model.device)
            with torch.no_grad():
                mis_prob = torch.softmax(mis_model.forward(mis_attack), dim=-1)[0]
                mis_prob = mis_prob.cpu().item()
            if (len(fighter_obs['mis_states'][np.where(fighter_obs['mis_states'] == 1)]) > 0) and (mis_prob < 0.9):
                fire_missile = 0
            elif (len(fighter_obs['mis_states'][np.where(fighter_obs['mis_states'] == 0)]) < 2) and (mis_prob < 0.9):
                fire_missile = 0
            else:
                fire_missile = int(mis_prob > 0.5)
        else:
            ac_model = self.opponent_list[self.opponent_id]['Pram,net']
            ac_model.eval()
            # print(ac_model.device)
            # 历史策略网络的对手
            # 将当前观测转化为网络输入
            obs_self_i = torch.as_tensor(fighter_obs['obs_self'], dtype=torch.float32, device=self.net_device)
            obs_target_i = torch.as_tensor(fighter_obs['obs_target'], dtype=torch.float32, device=self.net_device)
            mis_states_i = torch.as_tensor(fighter_obs['mis_states'], dtype=torch.int32, device=self.net_device)
            mis_alerts_i = torch.as_tensor(fighter_obs['mis_alerts'], dtype=torch.int32, device=self.net_device)

            # 在无梯度的情况下进行推理
            with torch.no_grad():
                _, a, _, _, _, self.rnn_hidden_state = ac_model.forward(obs_self_i, obs_target_i,
                                                                        mis_states_i, mis_alerts_i,
                                                                        hidden=self.rnn_hidden_state)
                act = a.cpu().numpy()
                load = 9 * act[0] if act[0] > 0 else (3 * act[0])
                load = min(9, max(-3, load))
                omega = min(300, max(-300, 300 * act[1]))
                rudder = min(1., max(-1., act[2]))
                thrust = min(1., max(0.1, 0.25 * act[3] + 0.75))
                mis_model.eval()
                mis_attack = torch.as_tensor(fighter_rnd_obs["mis_attack"], dtype=torch.float32,
                                             device=mis_model.device)
                mis_prob = torch.softmax(mis_model.forward(mis_attack), dim=-1)[0]
                mis_prob = mis_prob.cpu().item()
                if (len(fighter_obs['mis_states'][np.where(fighter_obs['mis_states'] == 1)]) > 0) and (mis_prob < 0.9):
                    fire_missile = 0
                elif (len(fighter_obs['mis_states'][np.where(fighter_obs['mis_states'] == 0)]) < 2) and (
                        mis_prob < 0.9):
                    fire_missile = 0
                else:
                    fire_missile = int(mis_prob > max(0.2, act[4] / 2 + 0.5))
                    # fire_missile = int(mis_prob > 0.5)

        if (fighter.fc_data.fAltitude < 500 + 5 * fighter.fc_data.fVerticalVelocity) and \
                (fighter.fc_data.fVerticalVelocity > 0):
            now_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
            safe_vec = [now_vec[0], now_vec[1], -0.7]

            thrust, load, omega, rudder = maneuver_lib.ppg.trainable_cmd(fighter.fc_data.fMachNumber,
                                                                         fighter.fc_data.fTrueAirSpeed,
                                                                         fighter.fc_data.fRollAngle,
                                                                         fighter.fc_data.fPitchAngle,
                                                                         fighter.fc_data.fYawAngle, safe_vec,
                                                                         maneuver_lib.decision_dt, [100, 0, 0], k_n=2.8)

        self.time_count += 1

        return thrust, load, omega, rudder, fire_missile

    def update_opponent_list(self, epoch, avg_proc_win_rate, avg_proc_op_win, policys_trained_len, history_index):
        if (epoch % 5 == 0) and (self.opponent_list[-1]['guidance']):
            if self.opponent_list[-1]['Pram,net'] > 0.4:
                for op_d in self.opponent_list:
                    op_d['guidance'] = True
                    op_d['Pram,net'] *= 0.99
                    op_d['index'] *= 0.99
            else:
                pass
        if ((avg_proc_win_rate / (avg_proc_op_win + 0.001)) > 1.05) and (epoch % 50 == 0):
            new_opponent_dict = {'guidance': False, 'Pram,net': None, 'index': 0}
            # 首先判断对手策略池中最后一个策略是制导还是网络
            if self.opponent_list[-1]['guidance']:
                # 判断制导的随机参数是否 < 1
                if self.opponent_list[-1]['Pram,net'] > 0.4:
                    new_opponent_dict['guidance'] = True
                    new_opponent_dict['Pram,net'] = self.opponent_list[-1]['Pram,net']
                    new_opponent_dict["index"] = self.opponent_list[-1]['index']
                else:
                    if (policys_trained_len > 1) and (self.self_play_index < history_index):
                        new_opponent_dict['guidance'] = False
                        new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "\\history_pi_net_" +
                                                                   str(self.self_play_index) + "_" +
                                                                   str(self.proc_id) + ".pt")
                        new_opponent_dict["index"] = self.self_play_index
                        self.self_play_index += 1
                    elif policys_trained_len > 1:
                        sfpid = np.random.randint(1, history_index, 1)[0]
                        new_opponent_dict['guidance'] = False
                        new_opponent_dict['Pram,net'] = torch.load(
                            self.self_play_dir + "\\history_pi_net_" + str(sfpid) + "_" + str(self.proc_id) + ".pt")
                        new_opponent_dict["index"] = sfpid
            else:
                if (policys_trained_len > 1) and (self.self_play_index < history_index):
                    new_opponent_dict['guidance'] = False
                    new_opponent_dict['Pram,net'] = torch.load(self.self_play_dir + "\\history_pi_net_" +
                                                               str(self.self_play_index) + "_" +
                                                               str(self.proc_id) + ".pt")
                    new_opponent_dict["index"] = self.self_play_index
                    self.self_play_index += 1
                elif policys_trained_len > 1:
                    sfpid = np.random.randint(1, history_index, 1)[0]
                    new_opponent_dict['guidance'] = False
                    new_opponent_dict['Pram,net'] = torch.load(
                        self.self_play_dir + "\\history_pi_net_" + str(sfpid) + "_" + str(self.proc_id) + ".pt")
                    new_opponent_dict["index"] = sfpid
            self.opponent_list = self.opponent_list[1:] + [new_opponent_dict]
            if all([opp['guidance'] == False for opp in self.opponent_list]):
                self.opponent_list[0]['guidance'] = True
                self.opponent_list[0]['Pram,net'] = 0.3
                self.opponent_list[0]['index'] = 0
