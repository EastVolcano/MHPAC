import argparse
import torch
import numpy as np
import os
import csv
import time
import datetime
import torch.distributed as t_dist
import torch.multiprocessing as mp
from PPO_mlp import PPOAgent
from SimSet.wvr_sim import make_sim_env, Reset
from Spinup.mpi_torch_utils import mpi_fork, setup_pytorch_for_mpi, proc_id, num_procs, \
    mpi_statistics_scalar, mpi_sum, mpi_avg
from Spinup.torch_distribute import statistics_scalar_torch
from train_logger import EpochLogger
from fighter_bot import ManeuversLib
from WVRENV.utils.data_record import record_tacview_outside
from WVRENV.utils.GNCData import wgs84ToNED, euler2vector, vector_angle
from Observation.featured_observation_V2_test import PartiallyObservations
from Reward.wvr_reward_V2 import reward_func, reward_baseline_1, reward_baseline_2
from Utils.util import RunningMeanStd
from ExploreAgent_V2 import RND
from Utils.DFOPS_V2_test import DFOPS
from scipy.spatial.transform import Rotation as R
from Utils.SECA import Engagement_wvr

# 决策时间步长 0.1 秒， 仿真步长默认0.01秒
Decision_dt = 0.1

lon0 = 160.123456
lat0 = 24.8976763
alt0 = 0.


def init_process(rank, size, pargs, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    t_dist.init_process_group(backend, init_method='tcp://localhost:64543', rank=rank, world_size=size,
                              timeout=datetime.timedelta(0, 180))
    fn(pargs)


def train(pargs):
    # SECA = Engagement_wvr(0)

    p_id = t_dist.get_rank() if pargs.torch_dist else proc_id()
    # 设置PPO推理、训练的device
    PPO_DEVICE = torch.device('cuda:' + str(p_id % 2))
    DFOPS_DEVICE = torch.device('cuda:' + str(p_id % 2))
    RND_DEVICE = torch.device('cuda:' + str(p_id % 2))

    # 初始化环境与输入量
    env, sim_in_list = make_sim_env(train=False)

    # set up Tacview dir and logger dir
    localtime = time.strftime("%Y_%m_%d", time.localtime())
    all_file_dir = ".\\output\\LogFiles\\" + pargs.exp_name + "_s" + str(pargs.seed)
    video_dir = os.path.join(all_file_dir, 'test_video_' + str(localtime))
    log_data_dir = os.path.join(all_file_dir, 'test_log_' + str(localtime))
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(log_data_dir, exist_ok=True)
    # Set up logger and save its configuration
    logger = EpochLogger(output_dir=log_data_dir, exp_name=pargs.exp_name, torch_dist=pargs.torch_dist)
    logger.save_config(locals())
    # set up file record for training
    process_num = t_dist.get_world_size() if pargs.torch_dist else num_procs()
    if p_id == 0:
        with (open(log_data_dir + '\\reward_data.csv', 'w') as csvfile):
            wirter = csv.writer(csvfile)
            fieldname = ['epoch'] + ['奖励_' + str(i) for i in range(len(env.world.fighters))] + ['奖励和'] + \
                        ['回合数'] + ['策略差异奖励'] + ['各进程平均胜率'] + [
                            '各进程对手平均胜率'] + ['各进程平均平局'] + \
                        ['各进程平均击杀率'] + ['各进程平均获胜时间'] + ['各进程rl平均剩余血量'] + [
                            '各进程对手平均剩余血量'] + ['命中概率拟合损失'] + ['命中概率拟合精度'] + \
                        ['RL平均首发导弹命中时间'] + ['对手平均首发导弹命中时间'] + ['mhp_loss']
            wirter.writerow(fieldname)
    # set up tacview output
    env.file_dir = video_dir

    if not pargs.torch_dist:
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

    # Random seed
    torch.manual_seed(pargs.seed + 1234 * p_id)
    torch.cuda.manual_seed(pargs.seed + 1234 * p_id)
    np.random.seed(pargs.seed + 1234 * p_id)

    # create rnd_ppo agents
    ppo_agents = []
    for i, fighter in enumerate(env.world.fighters):
        # act_dim: 4个飞机杆舵、油门控制量 + 1个导弹发射概率 输出范围均为[-1,1]
        if fighter.side == 0:
            ppo_agents.append(PPOAgent(i, 14, 8, 4, (4, 2),
                                       5, pargs, trainable=True, torch_dist=pargs.torch_dist, device=PPO_DEVICE))
        else:
            ppo_agents.append(PPOAgent(i, 14, 8, 4, (4, 2),
                                       5, pargs, trainable=False, torch_dist=pargs.torch_dist, device=PPO_DEVICE))
    # 加载训练的策略模型
    for i, agent in enumerate(ppo_agents):
        if agent.trainable:
            agent.actor_critic.actor.load_state_dict(torch.load(agent.save_dir + "pi_net.pt",
                                                                map_location='cuda:' + str(p_id % 2)))
            agent.actor_critic.actor.eval()
    # Create RND agents
    rnd_agents = []
    for i, fighter in enumerate(env.world.fighters):
        if fighter.side == 0:
            rnd_agents.append(RND(i, 9, pargs, trainable=True, torch_dist=pargs.torch_dist, device=RND_DEVICE))
        else:
            rnd_agents.append(RND(i, 9, pargs, trainable=False, torch_dist=pargs.torch_dist, device=RND_DEVICE))
    # 加载训练的导弹命中概率模型
    for i, r_agent in enumerate(rnd_agents):
        # if r_agent.trainable:
        r_agent.rnd_model.load_state_dict(torch.load(
            r_agent.save_dir + "missile_attack_" + str(0) + ".pt", map_location='cuda:' + str(p_id % 2)))
        r_agent.rnd_model.eval()

    # 建立观测对象
    agents_obs = []
    for i, fighter in enumerate(env.world.fighters):
        agents_obs.append(PartiallyObservations())

    # set up key data flags for log
    episode_num = 0
    op_ep_num = 0  # 用于记录对手策略池没有发生变化时的回合数
    r_n, d_n, ep_ret_n, ep_len = [0. for _ in range(len(env.world.fighters))], [False for _ in
                                                                                range(len(env.world.fighters))], \
        [0. for _ in range(len(env.world.fighters))], 0
    ep_ret_kl = [0. for _ in range(len(env.world.fighters))]  # 策略差异奖励
    # 被训练的智能体的胜率
    rl_win_num = 0
    avg_win_rate = 0
    # 对手胜率
    op_win_num = 0
    avg_op_win = 0
    # 被训练智能体的平局率
    rl_draw_num = 0
    avg_draw_rate = 0
    # 纯击杀率
    rl_sd_num = 0
    avg_sd_rate = 0
    # 获胜时间
    time_of_win = 0
    avg_tow = 0
    # 首发命中时间
    time_of_hit = 0
    avg_toh = 0
    time_of_hit_op = 0
    avg_toh_op = 0
    # 红蓝双方智能体每一局的血量
    rl_bloods = []
    op_bloods = []
    # mhp loss
    mhp_loss = []
    mis_avg_loss = 0
    mis_avg_acc = 0
    mis_data_num = 0

    # set up DFOPS
    # 生成机动库
    blue_maneuver_lib = ManeuversLib(Decision_dt)
    rl_maneuver_lib = ManeuversLib(Decision_dt)
    # 自博弈历史策略编号
    history_index = 0
    # 历史策略池
    policys_trained = []
    # 生成DFOPS对象
    df_ops = DFOPS(pargs.model_dir, pargs.rnn_type, (pargs.rnn_layer_num, pargs.rnn_hidden_size),
                   pargs.init_kar, p_id, net_device=DFOPS_DEVICE)

    # Main loop: collect experience in env and update/log each epoch
    local_steps_per_epoch = int(pargs.steps / process_num)
    record_flag = True

    # reset simulation env
    Reset(env)
    obs_n = []
    rnd_obs_n = []
    # 重置部分可观对象
    for o, obs_agent in enumerate(agents_obs):
        obs_agent.reset_obs_pram()
        if o == 0:
            obs_n.append(agents_obs[o].reshape_observation(env.world.fighters[0], env.world.fighters[1], env))
            rnd_obs_n.append(agents_obs[0].rnd_input(env.world.fighters[0], env.world.fighters[1], env))
        elif o == 1:
            obs_n.append(agents_obs[o].reshape_observation(env.world.fighters[1], env.world.fighters[0], env))
            rnd_obs_n.append(agents_obs[1].rnd_input(env.world.fighters[1], env.world.fighters[0], env))
        else:
            raise ValueError("智能体数量不对")

    # 重置RNN网络的隐状态（用于与环境交互推理）
    # rnn_hidden_state是当前推理所需要输入的RNN隐状态
    rnn_hidden_state_n = []
    for i_agent in range(len(env.world.fighters)):
        if pargs.rnn_type is None:
            rnn_hidden_state_n.append(None)
        elif pargs.rnn_type == 'lstm':
            rnn_hidden_state_n.append(
                (torch.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size), dtype=torch.float32).to(PPO_DEVICE),
                 torch.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size), dtype=torch.float32).to(PPO_DEVICE)))
        elif pargs.rnn_type == 'gru':
            rnn_hidden_state_n.append(
                torch.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size), dtype=torch.float32).to(PPO_DEVICE))
        else:
            raise ValueError("RNN网络形式设置错误")

    # 重置机动库
    rl_maneuver_lib.reset_lib(env.world.fighters[0], 0)
    blue_maneuver_lib.reset_lib(env.world.fighters[1], 0)
    # DFOPS采样
    df_ops.meta_solver()

    # 计时
    start_time = time.time()

    # 开始RL训练
    for epoch in range(pargs.epochs):
        terminal = False
        record_flag = True
        rl_bloods = []
        op_bloods = []
        mhp_loss = []

        # 单局的记录文件
        if (0 <= p_id <= 2) and (episode_num % 1 == 0):
            with open(log_data_dir + '\\paper_data_' + str(epoch) + '_' + str(proc_id()) + '.csv',
                      'w') as paperfile:
                p_wirter = csv.writer(paperfile)
                fieldname = ['time'] + ['red_lon'] + ['red_lat'] + ['red_alt'] + \
                            ['blue_lon'] + ['blue_lat'] + ['blue_alt'] + \
                            ['red_north'] + ['red_east'] + ['red_down'] + \
                            ['blue_north'] + ['blue_east'] + ['blue_down'] + \
                            ['red_roll'] + ['red_pitch'] + ['red_yaw'] + \
                            ['blue_roll'] + ['blue_pitch'] + ['blue_yaw'] + \
                            ['red_ATA'] + ['blue_ATA'] + \
                            ['red_AA'] + ['blue_AA'] + \
                            ['red_IAS'] + ['blue_IAS'] + \
                            ['red_energy'] + ['blue_energy'] + \
                            ['red_radius'] + ['blue_radius'] + \
                            ['red_rate'] + ['blue_rate'] + \
                            ['red_azcmd', 'red_aycmd', 'red_wxcmd'] + \
                            ['red_az', 'red_ay', 'red_wx'] + \
                            ['blue_az', 'blue_ay', 'blue_wx'] + \
                            ['red_blood', 'red_bullet', 'blue_blood', 'blue_bullet'] + \
                            ['red_mhp', 'blue_mhp', 'red_mhp_thread']
                p_wirter.writerow(fieldname)

        for t in range(pargs.max_ep_len + 1):

            action_n, rnn_hidden_n = [], []

            for i, agent in enumerate(ppo_agents):
                # 智能体与环境交互
                if agent.trainable:
                    # AC网络推理
                    obs_self_i = torch.as_tensor(obs_n[i]['obs_self'], dtype=torch.float32).to(PPO_DEVICE)
                    obs_target_i = torch.as_tensor(obs_n[i]['obs_target'], dtype=torch.float32).to(PPO_DEVICE)
                    mis_states_i = torch.as_tensor(obs_n[i]['mis_states'], dtype=torch.int32).to(PPO_DEVICE)
                    mis_alerts_i = torch.as_tensor(obs_n[i]['mis_alerts'], dtype=torch.int32).to(PPO_DEVICE)

                    # 在无梯度的情况下进行推理
                    with torch.no_grad():
                        a, rnn_hidden = agent.actor_critic.actor.get_forward(obs_self_i, obs_target_i,
                                                                             mis_states_i, mis_alerts_i,
                                                                             hidden=rnn_hidden_state_n[i])
                    action_n.append(a.cpu().numpy())

                    if pargs.rnn_type == 'lstm':
                        rnn_hidden_n.append((rnn_hidden[0].cpu().numpy(), rnn_hidden[1].cpu().numpy()))
                    elif pargs.rnn_type == 'gru':
                        rnn_hidden_n.append(rnn_hidden.cpu().numpy())
                    else:
                        pass
                else:
                    # 利用DFOPS采样的对手策略,生产控制指令
                    thrust, load, omega, rudder, fire_missile = (
                        df_ops.sampled_policy(env.world.fighters[1], env.world.fighters[0],
                                              blue_maneuver_lib, obs_n[1], ep_len,
                                              mis_model=rnd_agents[0].rnd_model, fighter_rnd_obs=rnd_obs_n[i]))
                    action_n.append(np.array([thrust, load, omega, rudder, fire_missile]))
                    if pargs.rnn_type == 'lstm':
                        rnn_hidden_n.append((np.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size),
                                                      dtype=np.float32),
                                             np.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size),
                                                      dtype=np.float32)))
                    else:
                        rnn_hidden_n.append(np.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size),
                                                     dtype=np.float32))

            # 飞机机动、导弹发射
            for i, agent in enumerate(ppo_agents):
                if agent.trainable:
                    load = 9 * action_n[i][0] if action_n[i][0] > 0 else (3 * action_n[i][0])
                    omega = 300 * action_n[i][1]
                    rudder = action_n[i][2]
                    thrust = 0.75 + 0.25 * action_n[i][3]

                    # ctrl = SECA.statemachine_wvr(fighter=env.world.fighters[0], enemy=env.world.fighters[1])
                    # thrust = ctrl[0]
                    # load = (ctrl[1] * 9) if ctrl[1] > 0 else (ctrl[1] * 3)
                    # omega = ctrl[2] * 300
                    # rudder = ctrl[3]

                    if env.world.fighters[i].sensors.alert_missile > 0:
                        thrust = 1.
                        # pass

                    # 高度保护
                    load_alt, omega_alt, rudder_alt = 0., 0., 0.
                    if (env.world.fighters[i].fc_data.fAltitude < 500 +
                        5 * env.world.fighters[i].fc_data.fVerticalVelocity) and \
                            (env.world.fighters[i].fc_data.fVerticalVelocity > 0):
                        now_vec = euler2vector(env.world.fighters[i].fc_data.fRollAngle,
                                               env.world.fighters[i].fc_data.fPitchAngle,
                                               env.world.fighters[i].fc_data.fYawAngle)
                        safe_vec = [now_vec[0], now_vec[1], -0.7]

                        thrust, load_alt, omega_alt, rudder_alt = rl_maneuver_lib.ppg.ctrl_cmd(
                            env.world.fighters[i].fc_data.fMachNumber,
                            env.world.fighters[i].fc_data.fTrueAirSpeed,
                            env.world.fighters[i].fc_data.fRollAngle,
                            env.world.fighters[i].fc_data.fPitchAngle,
                            env.world.fighters[i].fc_data.fYawAngle,
                            safe_vec,
                            Decision_dt)
                    load += 0.6 * load_alt
                    omega += 0.6 * omega_alt
                    rudder += 0.6 * rudder_alt

                    load = min(9, max(-3, load))
                    omega = min(300, max(-300, omega))
                    rudder = min(1., max(-1., rudder))
                    thrust = min(1., max(0.1, thrust))

                    # 导弹发射策略
                    rnd_agents[0].rnd_model.eval()
                    mis_attack_red = torch.as_tensor(rnd_obs_n[i]["mis_attack"], dtype=torch.float32, device=RND_DEVICE)
                    mis_prob_red = 0
                    with torch.no_grad():
                        mis_prob_red = torch.softmax(rnd_agents[0].rnd_model.forward(mis_attack_red), dim=-1)[0]
                        mis_prob_red = mis_prob_red.cpu().item()
                    if (len(obs_n[i]['mis_states'][np.where(obs_n[i]['mis_states'] == 1)]) > 0) and (mis_prob_red < 0.9):
                        fire_missile = 0
                    elif (len(obs_n[i]['mis_states'][np.where(obs_n[i]['mis_states'] == 0)]) < 2) and (mis_prob_red < 0.9):
                        fire_missile = 0
                    else:
                        fire_missile = int(mis_prob_red > max(0.2, action_n[i][4] / 2 + 0.5))
                        # fire_missile = int(mis_prob_red > 0.5)
                    red_mhp_thread = max(0.2, action_n[i][4] / 2 + 0.5)

                    # fire_missile = SECA.missile_fire_close
                    # fire_missile = 0
                    # 网络输出>0时不能发射，<=0时有可能发射
                else:
                    thrust = min(1., max(0.1, action_n[i][0]))
                    load = min(9., max(-3., action_n[i][1]))
                    omega = min(300., max(-300., action_n[i][2]))
                    rudder = min(1., max(-1., action_n[i][3]))
                    fire_missile = int(action_n[i][4])

                    mis_attack_blue = torch.as_tensor(rnd_obs_n[i]["mis_attack"], dtype=torch.float32, device=RND_DEVICE)
                    mis_prob_blue = 0
                    with torch.no_grad():
                        mis_prob_blue = torch.softmax(rnd_agents[0].rnd_model.forward(mis_attack_blue), dim=-1)[0]
                        mis_prob_blue = mis_prob_blue.cpu().item()

                sim_in_list[i].control_input = [thrust, (load / 9) if load > 0 else (load / 3), omega / 300, rudder]
                sim_in_list[i].missile_fire = fire_missile

            # 锁定目标
            sim_in_list[0].target_index = 1
            sim_in_list[1].target_index = 0

            # record tacview
            if 0 <= p_id <= 2:
                if (epoch % 1 == 0) and record_flag:
                    record_tacview_outside(video_dir, env, epoch, terminal, sim_in_list, p_id)

            if (0 <= p_id <= 2) and (episode_num % 1 == 0):
                red_n, red_e, red_d = wgs84ToNED(env.world.fighters[0].fc_data.fLatitude, env.world.fighters[0].fc_data.fLongitude,
                                                 env.world.fighters[0].fc_data.fAltitude, lat0, lon0, alt0)
                blue_n, blue_e, blue_d = wgs84ToNED(env.world.fighters[1].fc_data.fLatitude,
                                                    env.world.fighters[1].fc_data.fLongitude,
                                                    env.world.fighters[1].fc_data.fAltitude, lat0, lon0, alt0)
                ln, le, ld = wgs84ToNED(env.world.fighters[1].fc_data.fLatitude, env.world.fighters[1].fc_data.fLongitude,
                                        env.world.fighters[1].fc_data.fAltitude,
                                        env.world.fighters[0].fc_data.fLatitude, env.world.fighters[0].fc_data.fLongitude,
                                        env.world.fighters[0].fc_data.fAltitude)
                red_los = np.array([ln, le, ld])
                blue_los = np.array([-ln, -le, -ld])
                red_body_vec = euler2vector(env.world.fighters[0].fc_data.fRollAngle, env.world.fighters[0].fc_data.fPitchAngle,
                                            env.world.fighters[0].fc_data.fYawAngle)
                blue_body_vec = euler2vector(env.world.fighters[1].fc_data.fRollAngle, env.world.fighters[1].fc_data.fPitchAngle,
                                             env.world.fighters[1].fc_data.fYawAngle)
                red_ATA = vector_angle(red_los, red_body_vec)
                red_AA = vector_angle(red_los, blue_body_vec)
                blue_ATA = vector_angle(blue_los, blue_body_vec)
                blue_AA = vector_angle(blue_los, red_body_vec)
                red_rate = env.world.fighters[0].fc_data.fNormalLoad * 9.81 / (env.world.fighters[0].fc_data.fGroundSpeed + 0.001)
                blue_rate = env.world.fighters[1].fc_data.fNormalLoad * 9.81 / (env.world.fighters[1].fc_data.fGroundSpeed + 0.001)
                red_radius = env.world.fighters[0].fc_data.fGroundSpeed / (red_rate + 0.001)
                blue_radius = env.world.fighters[1].fc_data.fGroundSpeed / (blue_rate + 0.001)

                paper_data = []
                with (open(log_data_dir + '\\paper_data_' + str(epoch) + '_' + str(proc_id()) + '.csv', 'a',
                           newline='') as paperfile):
                    p_wirter_n = csv.writer(paperfile)
                    paper_data += [Decision_dt * ep_len] + [env.world.fighters[0].fc_data.fLongitude] + [
                        env.world.fighters[0].fc_data.fLatitude] + \
                                  [env.world.fighters[0].fc_data.fAltitude] + \
                                  [env.world.fighters[1].fc_data.fLongitude] + [env.world.fighters[1].fc_data.fLatitude] + [
                                      env.world.fighters[1].fc_data.fAltitude] + \
                                  [red_n] + [red_e] + [red_d] + \
                                  [blue_n] + [blue_e] + [blue_d] + \
                                  [env.world.fighters[0].fc_data.fRollAngle] + [env.world.fighters[0].fc_data.fPitchAngle] + [
                                      env.world.fighters[0].fc_data.fYawAngle] + \
                                  [env.world.fighters[1].fc_data.fRollAngle] + [env.world.fighters[1].fc_data.fPitchAngle] + [
                                      env.world.fighters[1].fc_data.fYawAngle] + \
                                  [red_ATA, blue_ATA] + [red_AA, blue_AA] + \
                                  [env.world.fighters[0].fc_data.fIndicatedAirSpeed,
                                   env.world.fighters[1].fc_data.fIndicatedAirSpeed] + \
                                  [env.world.fighters[0].fc_data.fAltitude + 0.5 * (
                                              env.world.fighters[0].fc_data.fGroundSpeed ** 2) / 9.81] + \
                                  [env.world.fighters[1].fc_data.fAltitude + 0.5 * (
                                          env.world.fighters[1].fc_data.fGroundSpeed ** 2) / 9.81] + \
                                  [red_radius, blue_radius, red_rate, blue_rate] + \
                                  [9 * action_n[0][0] if action_n[0][0] > 0 else (3 * action_n[0][0]), min(1, max(-1, action_n[0][2])),
                                   min(300, max(-300, 300 * action_n[0][1]))] + \
                                  [env.world.fighters[0].fc_data.fNormalLoad, env.world.fighters[0].fc_data.fLateralLoad,
                                   env.world.fighters[0].fc_data.fRollRate] + \
                                  [env.world.fighters[1].fc_data.fNormalLoad, env.world.fighters[1].fc_data.fLateralLoad,
                                   env.world.fighters[1].fc_data.fRollRate] + \
                                  [env.world.fighters[0].combat_data.bloods, env.world.fighters[0].combat_data.left_bullet,
                                   env.world.fighters[1].combat_data.bloods, env.world.fighters[1].combat_data.left_bullet] +\
                                  [mis_prob_red, mis_prob_blue, red_mhp_thread]
                    p_wirter_n.writerow(paper_data)

            # 一个决策步仿真更新
            for d_time in range(int(Decision_dt / 0.01)):
                missile_state_array_n = [
                    np.array([env.world.fighters[0].missiles[m].state
                              for m in range(len(env.world.fighters[0].missiles))]),
                    np.array([env.world.fighters[1].missiles[m].state
                              for m in range(len(env.world.fighters[1].missiles))])
                ]
                fire_state_n = [
                    agents_obs[0].rnd_input(env.world.fighters[0], env.world.fighters[1], env)['mis_attack'],
                    agents_obs[1].rnd_input(env.world.fighters[1], env.world.fighters[0], env)['mis_attack']
                ]

                terminal_flag = env.update(sim_in_list)

                # 更新rnd agent中的fire record
                missile_state_array_new_n = [
                    np.array([env.world.fighters[0].missiles[m].state
                              for m in range(len(env.world.fighters[0].missiles))]),
                    np.array([env.world.fighters[1].missiles[m].state
                              for m in range(len(env.world.fighters[1].missiles))])
                ]
                for ff, fighter in enumerate(env.world.fighters):
                    for mm in range(len(fighter.missiles)):
                        if missile_state_array_new_n[ff][mm] == 1 and missile_state_array_n[ff][mm] == 0:
                            rnd_agents[ff].fire_record['mis_fire_state'][mm] = fire_state_n[ff]
                        if missile_state_array_new_n[ff][mm] > 1 and missile_state_array_n[ff][mm] == 1:
                            rnd_agents[ff].fire_record['mis_state'][mm] = missile_state_array_new_n[ff][mm]

                # 计算第一发导弹命中的时刻
                if (len(missile_state_array_new_n[0][np.where(missile_state_array_new_n[0] == 2)])
                    > len(missile_state_array_n[0][np.where(missile_state_array_n[0] == 2)])) and \
                        (len(missile_state_array_new_n[0][np.where(missile_state_array_new_n[0] == 2)]) == 1):
                    time_of_hit += ep_len * Decision_dt + d_time * 0.01
                if (len(missile_state_array_new_n[1][np.where(missile_state_array_new_n[1] == 2)])
                    > len(missile_state_array_n[1][np.where(missile_state_array_n[1] == 2)])) and \
                        (len(missile_state_array_new_n[1][np.where(missile_state_array_new_n[1] == 2)]) == 1):
                    time_of_hit_op += ep_len * Decision_dt + d_time * 0.01

            # 获取下一步观测
            obs_next_n = [agents_obs[0].reshape_observation(env.world.fighters[0], env.world.fighters[1], env),
                          agents_obs[1].reshape_observation(env.world.fighters[1], env.world.fighters[0], env)]
            rnd_obs_next_n = [agents_obs[0].rnd_input(env.world.fighters[0], env.world.fighters[1], env),
                              agents_obs[1].rnd_input(env.world.fighters[1], env.world.fighters[0], env)]

            # 获取奖励
            r_n = [reward_func(rnd_agents[0].rnd_model,
                               env.world.fighters[0], obs_n[0], obs_next_n[0], rnd_obs_next_n[0],
                               obs_n[1], obs_next_n[1], rnd_obs_next_n[1],
                               env.world.fighters[1], env),
                   reward_func(rnd_agents[0].rnd_model,
                               env.world.fighters[1], obs_n[1], obs_next_n[1], rnd_obs_next_n[1],
                               obs_n[0], obs_next_n[0], rnd_obs_next_n[0],
                               env.world.fighters[0], env)]
            # 计算导弹命中概率拟合函数的损失 和 正确率
            with torch.no_grad():
                mis_loss = 0
                mis_acc = 0
                for ff, fighter in enumerate(env.world.fighters):
                    for mm in range(len(fighter.missiles)):
                        if obs_next_n[ff]['mis_states'][mm] == 2 and obs_n[ff]['mis_states'][mm] <= 1:
                            missile_attack_tensor = torch.as_tensor(rnd_agents[ff].fire_record['mis_fire_state'][mm],
                                                                    dtype=torch.float32, device=rnd_agents[0].device)
                            missile_label_tensor = torch.as_tensor([1, 0],
                                                                   dtype=torch.float32, device=rnd_agents[0].device)
                            pre = rnd_agents[0].rnd_model.forward(missile_attack_tensor)
                            mis_acc = torch.softmax(pre, dim=-1).argmax(dim=-1).eq(
                                missile_label_tensor.argmax(dim=-1)).sum().item()
                            mis_loss += rnd_agents[ff].compute_cross_entropy_error(missile_attack_tensor,
                                                                                   missile_label_tensor)
                            mis_avg_acc = (mis_avg_acc * mis_data_num + mis_acc) / (mis_data_num + 1)
                            mis_avg_loss = (mis_avg_loss * mis_data_num + mis_loss) / (mis_data_num + 1)
                            mis_data_num += 1

                            mhp_loss.append(rnd_agents[ff].compute_cross_entropy_error(missile_attack_tensor,
                                                                                           missile_label_tensor))

                        if obs_next_n[ff]['mis_states'][mm] == 3 and obs_n[ff]['mis_states'][mm] <= 1:
                            missile_attack_tensor = torch.as_tensor(rnd_agents[ff].fire_record['mis_fire_state'][mm],
                                                                    dtype=torch.float32, device=rnd_agents[0].device)
                            missile_label_tensor = torch.as_tensor([0, 1],
                                                                   dtype=torch.float32, device=rnd_agents[0].device)
                            pre = rnd_agents[0].rnd_model.forward(missile_attack_tensor)
                            mis_acc = torch.softmax(pre, dim=-1).argmax(dim=-1).eq(
                                missile_label_tensor.argmax(dim=-1)).sum().item()
                            mis_loss += rnd_agents[ff].compute_cross_entropy_error(missile_attack_tensor,
                                                                                   missile_label_tensor)
                            mis_avg_acc = (mis_avg_acc * mis_data_num + mis_acc) / (mis_data_num + 1)
                            mis_avg_loss = (mis_avg_loss * mis_data_num + mis_loss) / (mis_data_num + 1)
                            mis_data_num += 1

                            mhp_loss.append(rnd_agents[ff].compute_cross_entropy_error(missile_attack_tensor,
                                                                                       missile_label_tensor))
            # 确定done
            if pargs.missile_done:
                d_n = [False, False]
                for d in range(len(d_n)):
                    if env.world.fighters[d].combat_data.survive_info:
                        d_n[d] = False
                    else:
                        missile_state_array_next = obs_next_n[d]['mis_states']
                        if len(missile_state_array_next[np.where(missile_state_array_next == 1)]) > 0:
                            # 如果飞机死了，但它仍有发射的导弹在飞行，则不done
                            d_n[d] = False
                        else:
                            d_n[d] = True
            else:
                d_n = [False if env.world.fighters[0].combat_data.survive_info else True,
                       False if env.world.fighters[1].combat_data.survive_info else True]

            for i, agent in enumerate(ppo_agents):
                # 实际环境的累积奖励
                ep_ret_n[i] += r_n[i]

            # 更新观测！！！！!
            obs_n = obs_next_n
            rnd_obs_n = rnd_obs_next_n
            # 更新隐状态! ! ! !
            if pargs.rnn_type == 'lstm':
                for i_hidden in range(len(env.world.fighters)):
                    rnn_hidden_state_n[i_hidden] = (
                        torch.as_tensor(rnn_hidden_n[i_hidden][0], dtype=torch.float32).to(PPO_DEVICE),
                        torch.as_tensor(rnn_hidden_n[i_hidden][1], dtype=torch.float32).to(PPO_DEVICE)
                    )
            elif pargs.rnn_type == 'gru':
                for i_hidden in range(len(env.world.fighters)):
                    rnn_hidden_state_n = torch.as_tensor(rnn_hidden_n[i_hidden], dtype=torch.float32).to(PPO_DEVICE)
            else:
                pass

            ep_len += 1

            # 判断terminal
            timeout = (ep_len == pargs.max_ep_len)
            terminal = (terminal_flag >= 0) or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # record tacview at terminal
                if 0 <= p_id <= 2:
                    if (epoch % 1 == 0) and record_flag:
                        record_tacview_outside(video_dir, env, epoch, terminal, sim_in_list, p_id)
                # 记录终局血量
                rl_bloods.append(env.world.fighters[0].combat_data.bloods)
                op_bloods.append(env.world.fighters[1].combat_data.bloods)

                # 计算胜率
                episode_num += 1
                op_ep_num += 1
                if (env.world.fighters[0].combat_data.bloods > env.world.fighters[1].combat_data.bloods) and \
                        (env.world.fighters[0].combat_data.bloods > 0):
                    rl_win_num += 1
                    if (env.world.fighters[1].combat_data.bloods <= 0) and (
                            env.world.fighters[1].fc_data.fAltitude > 10):
                        rl_sd_num += 1
                    time_of_win += ep_len * Decision_dt
                elif (env.world.fighters[1].combat_data.bloods > env.world.fighters[0].combat_data.bloods) and \
                        (env.world.fighters[1].combat_data.bloods > 0):
                    op_win_num += 1
                    time_of_win += pargs.max_ep_len * Decision_dt
                else:
                    time_of_win += pargs.max_ep_len * Decision_dt
                    if (env.world.fighters[0].combat_data.bloods == env.world.fighters[1].combat_data.bloods) and \
                            (env.world.fighters[0].combat_data.bloods > 0):
                        rl_draw_num += 1
                # 计算首发命中时间
                if len(obs_next_n[0]['mis_states'][np.where(obs_next_n[0]['mis_states'] == 2)]) < 1:
                    time_of_hit += pargs.max_ep_len * Decision_dt
                if len(obs_next_n[1]['mis_states'][np.where(obs_next_n[1]['mis_states'] == 2)]) < 1:
                    time_of_hit_op += pargs.max_ep_len * Decision_dt

                avg_win_rate = rl_win_num / op_ep_num
                avg_op_win = op_win_num / op_ep_num
                avg_draw_rate = rl_draw_num / op_ep_num
                avg_sd_rate = rl_sd_num / op_ep_num
                avg_tow = time_of_win / op_ep_num
                avg_toh = time_of_hit / op_ep_num
                avg_toh_op = time_of_hit_op / op_ep_num

                # 重置，准备下一个episode
                if terminal:
                    record_flag = False
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet_n=np.sum(np.array(ep_ret_n[0])), EpLen=ep_len)
                    logger.store(Kl_Ret_n=np.sum(ep_ret_kl))
                    for ii in range(len(env.world.fighters)):
                        store_dict = {'EpRet_' + str(ii): ep_ret_n[ii]}
                        logger.store(**store_dict)

                # reset simulation env
                # SECA = Engagement_wvr(0)
                Reset(env)
                obs_n = []
                # 重置部分可观对象
                for o, obs_agent in enumerate(agents_obs):
                    obs_agent.reset_obs_pram()
                    if o == 0:
                        obs_n.append(
                            agents_obs[o].reshape_observation(env.world.fighters[0], env.world.fighters[1], env))
                    elif o == 1:
                        obs_n.append(
                            agents_obs[o].reshape_observation(env.world.fighters[1], env.world.fighters[0], env))

                # 重置RNN网络的隐状态（用于与环境交互推理）
                # rnn_hidden_state是当前推理所需要输入的RNN隐状态
                rnn_hidden_state_n = []
                for i_agent in range(len(env.world.fighters)):
                    if pargs.rnn_type is None:
                        rnn_hidden_state_n.append(None)
                    elif pargs.rnn_type == 'lstm':
                        rnn_hidden_state_n.append((
                            torch.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size), dtype=torch.float32).to(
                                PPO_DEVICE),
                            torch.zeros((pargs.rnn_layer_num, pargs.rnn_hidden_size), dtype=torch.float32).to(
                                PPO_DEVICE)
                        ))
                    elif pargs.rnn_type == 'gru':
                        rnn_hidden_state_n.append(
                            torch.zeros((
                                pargs.rnn_layer_num, pargs.rnn_hidden_size), dtype=torch.float32).to(PPO_DEVICE))

                # 重置机动库
                rl_maneuver_lib.reset_lib(env.world.fighters[0], 0)
                blue_maneuver_lib.reset_lib(env.world.fighters[1], 0)
                # DFOPS采样
                df_ops.meta_solver()

                ep_ret_n, ep_len = [0. for _ in range(len(env.world.fighters))], 0
                ep_ret_kl = [0. for _ in range(len(env.world.fighters))]

                break

        # save data for plot epret
        total_reward_data = logger.epoch_dict['EpRet_n']
        kl_reward_data = logger.epoch_dict['Kl_Ret_n']

        if pargs.torch_dist:
            total_reward_data, _ = statistics_scalar_torch(total_reward_data)
            total_reward_data = total_reward_data.cpu().item()

            kl_reward_data, _ = statistics_scalar_torch(kl_reward_data)
            kl_reward_data = kl_reward_data.cpu().item()

            # 各进程平均胜率
            avg_proc_win_rate, _ = statistics_scalar_torch([avg_win_rate])
            avg_proc_win_rate = avg_proc_win_rate.cpu().item()
            avg_proc_op_win, _ = statistics_scalar_torch([avg_op_win])
            avg_proc_op_win = avg_proc_op_win.cpu().item()
            avg_proc_draw_rate, _ = statistics_scalar_torch([avg_draw_rate])
            avg_proc_draw_rate = avg_proc_draw_rate.cpu().item()
            avg_proc_sd_rate, _ = statistics_scalar_torch([avg_sd_rate])
            avg_proc_sd_rate = avg_proc_sd_rate.cpu().item()
            avg_proc_tow, _ = statistics_scalar_torch([avg_tow])
            avg_proc_tow = avg_proc_tow.cpu().item()
            avg_proc_toh, _ = statistics_scalar_torch([avg_toh])
            avg_proc_toh = avg_proc_toh.cpu().item()
            avg_proc_toh_op, _ = statistics_scalar_torch([avg_toh_op])
            avg_proc_toh_op = avg_proc_toh_op.cpu().item()
            # RL智能体和对手各进程平均剩余血量
            avg_rl_bloods, _ = statistics_scalar_torch([np.mean(rl_bloods)])
            avg_rl_bloods = avg_rl_bloods.cpu().item()
            avg_op_bloods, _ = statistics_scalar_torch([np.mean(op_bloods)])
            avg_op_bloods = avg_op_bloods.cpu().item()

            # 导弹命中概率网络各进程平均拟合精度和损失函数
            avg_proc_mis_loss, _ = statistics_scalar_torch([mis_avg_loss])
            avg_proc_mis_loss = avg_proc_mis_loss.cpu().item()
            avg_proc_mis_acc, _ = statistics_scalar_torch([mis_avg_acc])
            avg_proc_mis_acc = avg_proc_mis_acc.cpu().item()

            avg_mhp_loss, _ = statistics_scalar_torch([np.mean(mhp_loss)])
            avg_mhp_loss = avg_mhp_loss.cpu().item()

        else:
            total_reward_data, _ = mpi_statistics_scalar(total_reward_data)
            kl_reward_data, _ = mpi_statistics_scalar(kl_reward_data)

            # 各进程平均胜率
            avg_proc_win_rate = mpi_avg(avg_win_rate)
            avg_proc_op_win = mpi_avg(avg_op_win)
            avg_proc_draw_rate = mpi_avg(avg_draw_rate)
            avg_proc_sd_rate = mpi_avg(avg_sd_rate)
            avg_proc_tow = mpi_avg(avg_tow)
            avg_proc_toh = mpi_avg(avg_toh)
            avg_proc_toh_op = mpi_avg(avg_toh_op)
            # RL智能体和对手各进程平均剩余血量
            avg_rl_bloods = mpi_avg(np.mean(rl_bloods))
            avg_op_bloods = mpi_avg(np.mean(op_bloods))

            # 导弹命中概率网络各进程平均拟合精度和损失函数
            avg_proc_mis_loss, _ = mpi_statistics_scalar([mis_avg_loss])
            avg_proc_mis_acc, _ = mpi_statistics_scalar([mis_avg_acc])

            avg_mhp_loss = mpi_avg(np.mean(mhp_loss))

        reward_data_n = [0. for _ in range(len(env.world.fighters))]
        for kk in range(len(env.world.fighters)):
            key = 'EpRet_' + str(kk)
            reward_data_n[kk] = logger.epoch_dict[key]
            logger.epoch_dict[key] = []
        for kk in range(len(env.world.fighters)):
            if pargs.torch_dist:
                reward_data_n[kk], _ = statistics_scalar_torch(reward_data_n[kk])
                reward_data_n[kk] = reward_data_n[kk].cpu().item()
            else:
                reward_data_n[kk], _ = mpi_statistics_scalar(reward_data_n[kk])

        if p_id == 0:
            print(f"epoch: {epoch}, reward: {reward_data_n}, rl_win_rate: {avg_proc_win_rate}, "
                  f"op_win_rate: {avg_proc_op_win}")

        train_data = []
        if p_id == 0:
            with (open(log_data_dir + '\\reward_data.csv', 'a', newline='') as csvfile):
                writer_n = csv.writer(csvfile)
                train_data += [epoch] + reward_data_n + [total_reward_data] + [episode_num] + \
                              [kl_reward_data] + [avg_proc_win_rate] + [avg_proc_op_win] + [avg_proc_draw_rate] + \
                              [avg_proc_sd_rate] + [avg_proc_tow] + \
                              [avg_rl_bloods] + [avg_op_bloods] + [avg_proc_mis_loss, avg_proc_mis_acc] + \
                              [avg_proc_toh, avg_proc_toh_op] + [avg_mhp_loss]
                writer_n.writerow(train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # exp set
    parser.add_argument('--seed', '-s', type=int, default=512)
    parser.add_argument('--exp_name', type=str, default='MHPAC_RPPO_10km_4_2')
    parser.add_argument('--model_dir', type=str, default='.\\output\\ActorCritic_test\\')
    parser.add_argument('--rnd_device', type=str, default='cuda:1')
    parser.add_argument('--ppo_device', type=str, default='cuda:0')
    parser.add_argument('--dfops_device', type=str, default='cuda:0')
    parser.add_argument('--torch_dist', type=bool, default=False)

    # sim set
    parser.add_argument('--max_ep_len', type=int, default=3000)
    parser.add_argument('--missile_done', type=bool, default=True)
    # training set
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--steps', type=int, default=100000)

    # rnn network parms in ppo
    parser.add_argument("-rt", "--rnn_type", default=None, type=str)
    parser.add_argument("-sl", "--seq_len", type=int, default=50)
    parser.add_argument("--rnn_layer_num", type=int, default=1)
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.99)
    parser.add_argument("--scheduled_opt_dmodel", type=float, default=50000,
                        help="pram for init lr in scheduled_opt")
    parser.add_argument("--warmup_steps", type=float, default=8000)

    # rnd parms
    parser.add_argument("-tw", "--task_weight", default=2.)
    parser.add_argument("-rw", "--rnd_weight", default=0.4)
    parser.add_argument("--gamma", type=float, default=0.993)
    parser.add_argument("--gamma_rnd", type=float, default=0.99)
    parser.add_argument("--rms_init_steps", type=int, default=1000)
    parser.add_argument("--predictor_lr", type=float, default=7e-4)

    # ppo parms
    parser.add_argument("--lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument("-kl", "--target_kl", type=float, default=0.05, help="kld constraint")
    parser.add_argument("--clip_ratio", type=float, default=0.1, help="clip ratio")
    parser.add_argument("--epoch_train_iters", type=int, default=10, help="iteration num in one epoch")
    parser.add_argument("--value_loss_weight", type=float, default=0.5, help="weight of value loss")

    # dfops parms
    parser.add_argument("--init_kar", type=float, default=0.15, help="initial kar of kar-PPG")

    pargs = parser.parse_args()

    if pargs.torch_dist:
        size = pargs.cpu
        processes = []
        mp.set_start_method("spawn")
        start_time = time.time()
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, pargs, train))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:

        mpi_fork(pargs.cpu)

        train(pargs)
