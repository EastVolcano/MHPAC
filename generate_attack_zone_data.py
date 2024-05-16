import csv
import os
import time
import numpy as np
from WVRENV.utils.GNCData import wgs84ToNED
from SimSet.missile_sim import make_sim_env, Reset
from Observation.featured_observation import observation, PartiallyObservations
from fighter_bot import ManeuversLib
from Spinup.mpi_torch_utils import mpi_fork, setup_pytorch_for_mpi, proc_id, num_procs, \
    mpi_statistics_scalar, mpi_sum, mpi_avg
from WVRENV.utils.data_record import record_tacview_outside


def generate_data(grid_set_dict=None):
    if grid_set_dict is None:
        grid_set_dict = {"alt_grid": np.linspace(1000, 10000, 5),  # 载机高度
                         "red_ma_grid": np.linspace(0.4, 1., 4),  # 载机马赫数
                         "fire_pitch_grid": np.linspace(-80, 80., 7),  # 载机俯仰角
                         "dist_grid": np.linspace(1000, 10000, 5),  # 目标距离
                         "body_q_t_grid": np.linspace(-60, 60., 5),  # 目标在安稳系下视线方位角
                         "body_q_d_grid": np.linspace(-60, 60., 5),  # 目标在安稳系下视线高低角
                         "blue_v_yaw_grid": np.linspace(-120, 180., 6),  # 目标航向
                         "blue_ma_grid": np.linspace(0.4, 1., 4)}  # 目标马赫数
    # 初始化环境与输入量
    env, sim_in_list = make_sim_env()

    # 计算仿真数
    sim_num = 1
    for key, grid in grid_set_dict.items():
        sim_num *= len(grid)

    # 记录所用变量
    red_fire, blue_fire = False, False
    red_missile_data = {"alt": 1000, "fire_ma": 0.8, "fire_pitch": 0, "target_dist": 2000, "body_q_t": 10,
                        "body_q_d": 0, "target_ma": 0.8, "target_hori_AA": 0, 'target_v_pitch': 0, 'hit': False}
    blue_missile_data = {"alt": 2000, "fire_ma": 0.8, "fire_pitch": 0, "target_dist": 2000, "body_q_t": 10,
                         "body_q_d": 0, "target_ma": 0.8, "target_hori_AA": 0, 'target_v_pitch': 0, 'hit': False}

    # 生成数据记录文件
    data_dir = '.\\DataSet\\missile_data\\'
    os.makedirs(data_dir, exist_ok=True)
    time_str = time.strftime("%Y_%m_%d", time.localtime())
    with open(data_dir + 'missile_data_' + str(proc_id()) + "_A_" + time_str + '.csv', 'w') as csvfile:
        wirter = csv.writer(csvfile)
        fieldname = [key for key, value in red_missile_data.items()]
        wirter.writerow(fieldname)

    # 生成蓝方（目标）机动库
    blue_maneuver_lib = ManeuversLib(env.initial_data.dt)
    # 生成红方机动库
    red_maneuver_lib = ManeuversLib(env.initial_data.dt)

    # 设置随机种子
    np.random.seed(104 * proc_id())

    start_time = time.time()
    # 开始批量仿真
    for i_episode in range(3000):
        # for i_episode in range(int((5 * sim_num) / num_procs())):
        # 设置仿真参数
        sim_set_dict = {"alt": grid_set_dict['alt_grid'][np.random.randint(len(grid_set_dict['alt_grid']))] +
                               np.random.uniform(-500, 500),  # 载机高度
                        "red_ma": grid_set_dict['red_ma_grid'][np.random.randint(len(grid_set_dict['red_ma_grid']))] +
                                  np.random.uniform(-0.1, 0.1),  # 载机马赫数
                        "fire_pitch": grid_set_dict['fire_pitch_grid'][
                                          np.random.randint(len(grid_set_dict['fire_pitch_grid']))] +
                                      np.random.uniform(-9, 9),  # 载机俯仰角
                        "dist": grid_set_dict['dist_grid'][np.random.randint(len(grid_set_dict['dist_grid']))] +
                                np.random.uniform(-500, 500),  # 目标距离
                        "body_q_t": grid_set_dict['body_q_t_grid'][
                                        np.random.randint(len(grid_set_dict['body_q_t_grid']))] +
                                    np.random.uniform(-10, 10),  # 目标在安稳系下视线方位角
                        "body_q_d": grid_set_dict['body_q_d_grid'][
                                        np.random.randint(len(grid_set_dict['body_q_d_grid']))] +
                                    np.random.uniform(-10, 10),  # 目标在安稳系下视线高低角
                        "blue_v_yaw": grid_set_dict['blue_v_yaw_grid'][
                                          np.random.randint(len(grid_set_dict['blue_v_yaw_grid']))] +
                                      np.random.uniform(-15, 15),  # 目标航向
                        "blue_ma": grid_set_dict['blue_ma_grid'][
                                       np.random.randint(len(grid_set_dict['blue_ma_grid']))] +
                                   np.random.uniform(-0.1, 0.1)  # 目标马赫数
                        }
        # sim_set_dict = {"alt": 8469,  # 载机高度
        #                 "red_ma": 0.969,  # 载机马赫数
        #                 "fire_pitch": 82.7,  # 载机俯仰角
        #                 "dist": 8000,  # 目标距离
        #                 "body_q_t": -0,  # 目标在安稳系下视线方位角
        #                 "body_q_d": 90,  # 目标在安稳系下视线高低角
        #                 "blue_v_yaw": -8,  # 目标航向
        #                 "blue_ma": 1.04  # 目标马赫数
        #                 }
        # 重置仿真
        Reset(env, sim_set_dict)
        # print(f"episode: {i_episode}, sim_set_alt: {sim_set_dict['alt']}, env_alt: {env.world.fighters[0].fc_data.fAltitude}")
        # 重置蓝方机动库
        blue_maneuver_lib.reset_lib(env.world.fighters[1], np.random.randint(len(blue_maneuver_lib)))
        # 重置红方机动库
        red_maneuver_lib.reset_lib(env.world.fighters[0], 4 + np.random.randint(3))
        # blue_maneuver_lib.reset_lib(env.world.fighters[1], 0)
        # print(f"episode: {i_episode}, blue maneuver: {blue_maneuver_lib.maneuver_index}")
        # if proc_id() == 22:
        #     print(f"episode: {i_episode}, maneuver_index : {blue_maneuver_lib.maneuver_index}")
        # 重置部分可观对象
        red_pobs, blue_pobs = PartiallyObservations(), PartiallyObservations()

        # 重置记录所用变量
        red_missile_data['hit'] = False
        blue_missile_data['hit'] = False
        red_fire, blue_fire = False, False
        red_flying, blue_flying = True, True
        redm_time, bluem_time = 0, 0  # 导弹的工作时间
        red_surv_time, blue_surv_time = 0, 0  # 飞机意外坠毁的时间

        # 开始单轮仿真
        for t in range(20000):

            # if proc_id() == 22 or proc_id() == 0:
            #     record_tacview_outside("E:\\ReinforcementLearning\\RNDPPO\\DataSet\\", env, i_episode, False, proc_id())
            # if i_episode == 0:
            #     print(f"time: {t}, fire_data: {blue_missile_data}")

            # 获取观测数据
            red_obs = observation(env.world.fighters[0], env.world.fighters[1], env)
            blue_obs = observation(env.world.fighters[1], env.world.fighters[0], env)
            # 编辑输入控制数据
            # 飞机机动
            if np.random.uniform(0, 1) > 0.3:
                red_thrust, red_load, red_omega, red_rudder = red_maneuver_lib.step_decision(env.world.fighters[0],
                                                                                             env.world.fighters[1])
                sim_in_list[0].control_input = [red_thrust, red_load / 9 if red_load >= 0 else red_load / 3,
                                                red_omega / 300, red_rudder]
            else:
                sim_in_list[0].control_input = [1, 1 / 9, 0, 0]

            blue_thrust, blue_load, blue_omega, blue_rudder = blue_maneuver_lib.step_decision(env.world.fighters[1],
                                                                                              env.world.fighters[0])
            sim_in_list[1].control_input = [blue_thrust, blue_load / 9 if blue_load >= 0 else blue_load / 3,
                                            blue_omega / 300, blue_rudder]

            # 发射导弹 红方开局5秒内随机选择时刻发射，蓝方满足条件可发射
            sim_in_list[0].target_index = 1
            red_fire_time = np.random.randint(0, 500)
            if t > red_fire_time:
                sim_in_list[0].missile_fire = 1
            sim_in_list[1].target_index = 0
            sim_in_list[1].missile_fire = 1

            # 单个回合更新
            terminal = env.update(sim_in_list)
            # 更新观测数据
            red_obs_next = observation(env.world.fighters[0], env.world.fighters[1], env)
            blue_obs_next = observation(env.world.fighters[1], env.world.fighters[0], env)

            # 实时可视化更新
            env.tcp_update(t)

            # 记录导弹发射和命中状态
            if red_obs_next["discrete"]["missiles_self_state"][0] > 0:
                red_fire = True
            else:
                red_fire = False
            if blue_obs_next["discrete"]["missiles_self_state"][0] > 0:
                blue_fire = True
            else:
                blue_fire = False

            # 记录导弹是否在飞行中
            if red_obs_next["discrete"]["missiles_self_state"][0] > 1:
                red_flying = False
            if blue_obs_next["discrete"]["missiles_self_state"][0] > 1:
                blue_flying = False

            if red_obs_next["discrete"]["missiles_self_state"][0] <= 1:
                redm_time += 1
            if env.world.fighters[0].combat_data.be_effective_killed:
                red_surv_time += 1
            if blue_obs_next["discrete"]["missiles_self_state"][0] <= 1:
                bluem_time += 1
            if env.world.fighters[1].combat_data.be_effective_killed:
                blue_surv_time += 1

            if (red_obs_next["discrete"]["missiles_self_state"][0] == 1) and \
                    (red_obs["discrete"]["missiles_self_state"][0] == 0):
                red_missile_data['alt'] = red_obs["continuous"]["missile_attack_zone"]['alt_self']
                red_missile_data['fire_ma'] = red_obs["continuous"]["missile_attack_zone"]['ma_self']
                red_missile_data['fire_pitch'] = red_obs["continuous"]["missile_attack_zone"]['pitch_self']
                red_missile_data['target_dist'] = red_obs["continuous"]["missile_attack_zone"]['dist']
                red_missile_data['target_v_pitch'] = red_obs["continuous"]["missile_attack_zone"]['target_v_pitch']
                red_missile_data['target_ma'] = red_obs["continuous"]["missile_attack_zone"]['ma_target']
                red_missile_data['target_hori_AA'] = red_obs["continuous"]["missile_attack_zone"]['AA_hori']
                red_missile_data['body_q_t'] = red_obs["continuous"]["missile_attack_zone"]['los_yaw_body']
                red_missile_data['body_q_d'] = red_obs["continuous"]["missile_attack_zone"]['los_pitch_body']

                # red_missile_data['alt'] = red_obs_next["continuous"]["missile_attack_zone"]['alt_self']
                # red_missile_data['fire_ma'] = red_obs_next["continuous"]["missile_attack_zone"]['ma_self']
                # red_missile_data['fire_pitch'] = red_obs_next["continuous"]["missile_attack_zone"]['pitch_self']
                # red_missile_data['target_dist'] = red_obs_next["continuous"]["missile_attack_zone"]['dist']
                # red_missile_data['target_v_pitch'] = red_obs_next["continuous"]["missile_attack_zone"]['target_v_pitch']
                # red_missile_data['target_ma'] = red_obs_next["continuous"]["missile_attack_zone"]['ma_target']
                # red_missile_data['target_hori_AA'] = red_obs_next["continuous"]["missile_attack_zone"]['AA_hori']
                # red_missile_data['body_q_t'] = red_obs_next["continuous"]["missile_attack_zone"]['los_yaw_body']
                # red_missile_data['body_q_d'] = red_obs_next["continuous"]["missile_attack_zone"]['los_pitch_body']

            if (blue_obs_next["discrete"]["missiles_self_state"][0] == 1) and \
                    (blue_obs["discrete"]["missiles_self_state"][0] == 0):
                blue_missile_data['alt'] = blue_obs["continuous"]["missile_attack_zone"]['alt_self']
                blue_missile_data['fire_ma'] = blue_obs["continuous"]["missile_attack_zone"]['ma_self']
                blue_missile_data['fire_pitch'] = blue_obs["continuous"]["missile_attack_zone"]['pitch_self']
                blue_missile_data['target_dist'] = blue_obs["continuous"]["missile_attack_zone"]['dist']
                blue_missile_data['target_v_pitch'] = blue_obs["continuous"]["missile_attack_zone"][
                    'target_v_pitch']
                blue_missile_data['target_ma'] = blue_obs["continuous"]["missile_attack_zone"]['ma_target']
                blue_missile_data['target_hori_AA'] = blue_obs["continuous"]["missile_attack_zone"]['AA_hori']
                blue_missile_data['body_q_t'] = blue_obs["continuous"]["missile_attack_zone"]['los_yaw_body']
                blue_missile_data['body_q_d'] = blue_obs["continuous"]["missile_attack_zone"]['los_pitch_body']

                # blue_missile_data['alt'] = blue_obs_next["continuous"]["missile_attack_zone"]['alt_self']
                # blue_missile_data['fire_ma'] = blue_obs_next["continuous"]["missile_attack_zone"]['ma_self']
                # blue_missile_data['fire_pitch'] = blue_obs_next["continuous"]["missile_attack_zone"]['pitch_self']
                # blue_missile_data['target_dist'] = blue_obs_next["continuous"]["missile_attack_zone"]['dist']
                # blue_missile_data['target_v_pitch'] = blue_obs_next["continuous"]["missile_attack_zone"][
                #     'target_v_pitch']
                # blue_missile_data['target_ma'] = blue_obs_next["continuous"]["missile_attack_zone"]['ma_target']
                # blue_missile_data['target_hori_AA'] = blue_obs_next["continuous"]["missile_attack_zone"]['AA_hori']
                # blue_missile_data['body_q_t'] = blue_obs_next["continuous"]["missile_attack_zone"]['los_yaw_body']
                # blue_missile_data['body_q_d'] = blue_obs_next["continuous"]["missile_attack_zone"]['los_pitch_body']

            if red_obs_next["discrete"]["missiles_self_state"][0] == 2:
                red_missile_data['hit'] = True
            else:
                red_missile_data['hit'] = False

            if blue_obs_next["discrete"]["missiles_self_state"][0] == 2:
                blue_missile_data['hit'] = True
            else:
                blue_missile_data['hit'] = False

            # 仿真自然结束
            if (terminal >= 0) or (red_obs_next["discrete"]["missiles_self_state"][0] == 3 and
                                   blue_obs_next["discrete"]["missiles_self_state"][0] == 3):
                # print(f"episode: {i_episode} 结束")
                break
            # 仿真强制结束 1: 时间一分钟 ，没有导弹在飞行中
            if ((t > 6000) and (not red_obs_next["discrete"]["missiles_self_state"][0] == 1) and
                    (not blue_obs_next["discrete"]["missiles_self_state"][0] == 1)):
                # print(f"episode: {i_episode} 强制结束 1")
                break

        # 将记录的数据写文件
        with open(data_dir + 'missile_data_' + str(proc_id()) + "_A_" + time_str + '.csv', 'a', newline='') as csvfile:
            writer_n = csv.writer(csvfile)
            # 只有当开火并, 且仿真结束时导弹不处于飞行中才会记录，且目标没有在被命中或者导弹失效前坠毁
            if red_fire and (not red_flying) and (blue_surv_time >= redm_time):
                writer_data_red = [value for key, value in red_missile_data.items()]
                writer_n.writerow(writer_data_red)
            if blue_fire and (not blue_flying) and (red_surv_time >= bluem_time):
                writer_data_blue = [value for key, value in blue_missile_data.items()]
                writer_n.writerow(writer_data_blue)

        if proc_id() == 0:
            print(f"now episode is {i_episode}, used time: {time.time() - start_time} sec")


if __name__ == '__main__':
    mpi_fork(50)

    generate_data()
