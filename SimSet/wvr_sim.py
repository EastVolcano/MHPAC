from WVRENV_PHD.SimArg import InitialData
from WVRENV_PHD.SimInput import FighterDataIn
from WVRENV_PHD.simulation_env import CombatEnv
from WVRENV_PHD.utils.GNCData import vector_angle, euler2vector, ned_to_body
from scipy.spatial.transform import Rotation as R
import numpy as np


def make_sim_env(train=True):
    """
    :return: env 仿真环境实例 sim_in_list 仿真环境中所有飞机的控制输入实例
    """
    # 实例化环境
    env = CombatEnv()
    # 环境初始数据
    initial_data = InitialData()
    initial_data.train_flag = train  # 是否是RL训练模式
    initial_data.log_tacview = False  # 是否开启内置的仿真文件记录功能
    initial_data.log_csv = False  # 是否输出记录文件
    initial_data.dll_str = '.\\MultiFighter.dll'  # 调用的模型路径，若加载失败，可改为绝对路径

    # 仿真设定
    initial_data.dt = 0.01  # 最小数据更新率（底层步长为0.01s）
    initial_data.len_max = 30000  # 单轮仿真长度

    # 红蓝双方战机数量
    initial_data.num_blue = 1  # 蓝机数量
    initial_data.num_red = 1  # 红机数量
    initial_data.originLongitude = 160.123456  # 仿真的经纬度原点位置
    initial_data.originLatitude = 24.8976763

    # 初始载弹量
    initial_data.missiles_max = 2

    # 机载雷达范围设定
    initial_data.radar_range = 40000
    initial_data.radar_vertical_scan = 30  # 雷达垂直扫描范围
    initial_data.radar_horizontal_scan = 30  # 雷达水平扫描范围
    initial_data.eodas_range = 10000  # 光电分布式探测孔径系统（EODAS）探测范围
    initial_data.alert_missile_range = 4000  # 来袭导弹告警范围
    initial_data.missile_without_radar = True  # 导弹发射是否不依赖雷达锁定的开关

    sim_in_list = [FighterDataIn() for _ in range(initial_data.num_blue + initial_data.num_red)]

    # 飞机设定控制模式
    for i in range(initial_data.num_blue + initial_data.num_red):
        sim_in_list[i].control_mode = 0

    # 完成初始化
    env.initial(sim_in_list, initial_data)

    return env, sim_in_list


def Reset(env, sim_set_dict=None):
    """
    复写仿真环境的重置功能
    :param env: 仿真环境实例
    :param env: 仿真环境初始化数据实例
    :param sim_set_dict: 仿真想定参数
    """
    if sim_set_dict is None:
        # 载机高度、马赫数、导弹发射俯仰角 # 目标距离、安稳系下视线方位角、高低角 # 目标航向、马赫数
        sim_set_dict = {"alt": np.random.uniform(3000, 10000), "red_ma": np.random.uniform(0.7, 1.2),
                        "fire_pitch": 0,  "dist": np.random.uniform(10000, 20000),
                        "body_q_t": np.random.uniform(-60, 60), "body_q_d": np.random.uniform(-20, 20),
                        "blue_v_yaw": np.random.uniform(-180, 180),
                        "blue_ma": np.random.uniform(0.7, 1.2)}

    # 重置载机和目标的航向
    red_orientation = np.random.uniform(-180, 180)
    blue_orientation = red_orientation + 180 + np.random.uniform(-50, 50)
    env.initial_data.orientation = []
    env.initial_data.orientation.append(red_orientation)
    env.initial_data.orientation.append(blue_orientation)

    # 重置载机和目标位置
    env.initial_data.NED = []
    # 载机位于NED原点，高度变化
    for i in range(env.num_RedFighter):
        env.initial_data.NED.append([0, 0, -sim_set_dict['alt']])

    # 计算目标载NED系下的位置
    los_stable_body = euler2vector(0, sim_set_dict['body_q_d'], sim_set_dict['body_q_t'])
    r_ned2body = R.from_euler('ZYX', [red_orientation, 0, 0], degrees=True).inv()
    R_StabbleBody2NED = r_ned2body.inv().as_matrix()
    los_NED = np.matmul(R_StabbleBody2NED, los_stable_body)
    for i in range(env.num_BlueFighter):
        env.initial_data.NED.append([0 + los_NED[0] * sim_set_dict['dist'], 0 + los_NED[1] * sim_set_dict['dist'],
                                     max(min(-sim_set_dict['alt'] + los_NED[2] * sim_set_dict['dist'], -2000), -10000)])

    # 重置载机和目标的马赫数
    env.initial_data.ma = []
    for i in range(env.num_RedFighter):
        env.initial_data.ma.append(sim_set_dict['red_ma'])
    for i in range(env.num_BlueFighter):
        env.initial_data.ma.append(sim_set_dict['blue_ma'])

    # 重置油量
    env.initial_data.FuelKg = []
    for i in range(env.num_RedFighter + env.num_BlueFighter):
        env.initial_data.FuelKg.append(0.8 * 3000)

    # 调用reset函数
    env.reset()
