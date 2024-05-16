"""
将环境中的信息转化智能体的观测量，同时适用于攻击区拟合网络的输入
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from WVRENV_PHD.utils.GNCData import wgs84ToNED, euler2vector, vector_angle, ned_to_body


def discrete_alert(x, num_bin):
    if x > 0.999:
        xd = num_bin
    elif x < -0.999:
        xd = 1
    else:
        xd = int(np.floor((x - (-1)) / (2 / num_bin))) + 1
    return xd


def logistic(x, growth_rate, mid_point):
    if (-growth_rate * (x - mid_point)) > 700:
        S = 0
    else:
        S = 1 / (1 + np.exp(-growth_rate * (x - mid_point)))
    return S


def observation(fighter, target, env):
    """
    :param fighter: 智能体对象
    :param target: 战斗机观测的目标
    :param env: WVRENV_PHD
    :return: obs字典：分连续观测和离散观测, 连续观测中区分导弹发射相关状态、和非导弹发射相关状态
    """
    obs_dict = {"continuous": {}, "discrete": {}}

    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
    los_vec = np.array([l_n, l_e, l_d])
    # 机体系的目标距离、高低角、方位角
    los_vec_body = ned_to_body(los_vec, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                               fighter.fc_data.fRollAngle)

    # 视线矢量在机体系下的俯仰角和偏航角
    los_pitch_body = np.rad2deg(-np.arctan2(los_vec_body[2], np.linalg.norm(los_vec_body[0:2])))
    los_yaw_body = np.rad2deg(np.arctan2(los_vec_body[1], los_vec_body[0]))
    dist = np.linalg.norm(los_vec)

    # 目标水平面内的进入角 (-180~180, 目标速度相对视线向右偏为正)
    AA_hori = vector_angle([l_n, l_e, 0], [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, 0])
    AA_hori *= np.sign(np.cross([l_n, l_e], [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity]))

    # 目标接近率
    vel = np.array([fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
    vel_t = np.array([target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
    if dist > 1:
        dist_dot = np.linalg.norm(vel_t) * \
                   (vel_t.dot(los_vec) / (np.linalg.norm(vel_t) * dist)) - \
                   np.linalg.norm(vel) * \
                   (vel.dot(los_vec) / (np.linalg.norm(vel) * dist))
    else:
        dist_dot = 0
    # 目标三维视线偏置角
    ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
    ATA = vector_angle(ati_vec, los_vec)

    obs_dict["continuous"]["missile_attack_zone"] = {"dist": dist, "los_yaw_body": los_yaw_body,
                                                     "los_pitch_body": los_pitch_body, "AA_hori": AA_hori,
                                                     "pitch_self": fighter.fc_data.fPitchAngle,  # 自身俯仰角
                                                     "target_v_pitch": target.fc_data.fPathPitchAngle,  # 目标爬升角
                                                     "alt_self": fighter.fc_data.fAltitude,  # 自身高度
                                                     "ma_self": fighter.fc_data.fMachNumber,  # 自身马赫数
                                                     "ma_target": target.fc_data.fMachNumber}  # 目标马赫数
    obs_dict["continuous"]["others"] = {"roll_self": fighter.fc_data.fRollAngle,  # 自身滚转角
                                        "roll_rate_self": fighter.fc_data.fRollRate,  # 自身滚转速率
                                        "normal_load_self": fighter.fc_data.fNormalLoad,  # 自身法向过载
                                        "lateral_load_self": fighter.fc_data.fLateralLoad,  # 自身侧向过载
                                        "fuel_self": fighter.fc_data.fNumberofFuel,  # 自身油量
                                        "dist_dot": dist_dot, "ATA": ATA}

    # 假设每架战斗机最多携带8枚导弹，每枚导弹的状态分为：0 未发射；1 飞行中；2 命中；3 失效，如果总数不足8枚，则不足的部分全为失效状态
    if len(fighter.missiles) >= 8:
        missiles_self_state_obs = [fighter.missiles[m].state for m in range(8)]
    else:
        missiles_self_state_obs = [fighter.missiles[m].state for m in range(len(fighter.missiles))]
        missiles_self_state_obs += [3 for m in range(8 - len(fighter.missiles))]

    # 将导弹告警进行符号化表示，将机体系下的球坐标划分为18个关于机体纵轴对称的区域，告警方向与机体纵轴三维夹角每10°是一个区域
    # 每个区域又分四个象限 高低、方位均为正是第一象限，方位为正、高低为负是第二象限，方位、高低均为负是第三象限
    # 方位为负、高低为正是第四象限；
    # 若一个区域内不同象限均有告警，则都记录在该区域的列表中，同一象限的只计一次
    # 0 表示该象限没有告警，1 表示有
    missile_alert_tokens_obs = [[0, 0, 0, 0] for _ in range(18)]
    for m_alert_id in range(fighter.sensors.alert_missile):
        # 计算告警所在区域
        alert_vec_body = euler2vector(0, fighter.sensors.alert_pitch[m_alert_id],
                                      fighter.sensors.alert_yaw[m_alert_id])
        alert_angle_3d = vector_angle([1, 0, 0], alert_vec_body)
        alert_token_id = int(np.floor(alert_angle_3d / 10)) if alert_angle_3d < 179 else 17

        # 计算告警对应区域的象限
        if (fighter.sensors.alert_pitch[m_alert_id] >= 0) and (fighter.sensors.alert_yaw[m_alert_id] >= 0):
            missile_alert_tokens_obs[alert_token_id][0] = 1
        elif (fighter.sensors.alert_pitch[m_alert_id] < 0) and (fighter.sensors.alert_yaw[m_alert_id] > 0):
            missile_alert_tokens_obs[alert_token_id][1] = 1
        elif (fighter.sensors.alert_pitch[m_alert_id] < 0) and (fighter.sensors.alert_yaw[m_alert_id] < 0):
            missile_alert_tokens_obs[alert_token_id][2] = 1
        elif (fighter.sensors.alert_pitch[m_alert_id] > 0) and (fighter.sensors.alert_yaw[m_alert_id] < 0):
            missile_alert_tokens_obs[alert_token_id][3] = 1

    obs_dict["discrete"]["missiles_self_state"] = missiles_self_state_obs
    obs_dict["discrete"]["missile_alert_tokens"] = missile_alert_tokens_obs

    return obs_dict


class PartiallyObservations(object):
    def __init__(self):
        self.dist = 0
        self.los_pitch_body = 0
        self.los_yaw_body = 0

    def reset_obs_pram(self):
        self.dist = 0
        self.los_pitch_body = 0
        self.los_yaw_body = 0

    def partially_observe(self, fighter, target, env):
        """
        依据传感器探测和态势预警给定观测
        :param fighter: 智能体对象
        :param target: 智能体观测目标
        :param env: WVRENV_PHD
        :return: obs字典：分连续观测和离散观测, 连续观测中区分导弹发射相关状态、和非导弹发射相关状态
        """
        obs_dict = {"continuous": {}, "discrete": {}}

        # 传感器约束
        if not (target.index in fighter.sensors.eodas_list):
            if env.time_count % 500 == 0:
                l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude,
                                           target.fc_data.fAltitude,
                                           fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                                           fighter.fc_data.fAltitude)
                los_vec = np.array([l_n, l_e, l_d])
                # 机体系的目标距离、高低角、方位角
                los_vec_body = ned_to_body(los_vec, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                                           fighter.fc_data.fRollAngle)

                # 视线矢量在安稳坐标系下的俯仰角和偏航角
                self.los_pitch_body = np.rad2deg(-np.arctan2(los_vec_body[2], np.linalg.norm(los_vec_body[0:2])))
                self.los_yaw_body = np.rad2deg(np.arctan2(los_vec_body[1], los_vec_body[0]))
                self.dist = np.linalg.norm(los_vec)
            else:
                pass
            AA_hori, dist_dot, ATA, target_v_pitch, tg_ma = 0, 0, 0, 0, 0
        else:
            l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                                       fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
            los_vec = np.array([l_n, l_e, l_d])
            # 机体系的目标距离、高低角、方位角
            los_vec_body = ned_to_body(los_vec, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                                       fighter.fc_data.fRollAngle)

            # 视线矢量在机体坐标系下的俯仰角和偏航角
            self.los_pitch_body = np.rad2deg(-np.arctan2(los_vec_body[2], np.linalg.norm(los_vec_body[0:2])))
            self.los_yaw_body = np.rad2deg(np.arctan2(los_vec_body[1], los_vec_body[0]))
            self.dist = np.linalg.norm(los_vec)

            # 目标水平面内的进入角 (-180~180, 目标速度相对视线向右偏为正)
            AA_hori = vector_angle([l_n, l_e, 0], [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, 0])
            AA_hori *= np.sign(np.cross([l_n, l_e], [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity]))

            # 目标接近率
            vel = np.array(
                [fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
            vel_t = np.array(
                [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
            if self.dist > 1:
                dist_dot = np.linalg.norm(vel_t) * \
                           (vel_t.dot(los_vec) / (np.linalg.norm(vel_t) * self.dist)) - \
                           np.linalg.norm(vel) * \
                           (vel.dot(los_vec) / (np.linalg.norm(vel) * self.dist))
            else:
                dist_dot = 0
            # 目标三维视线偏置角
            ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
            ATA = vector_angle(ati_vec, los_vec)
            # 目标爬升角
            target_v_pitch = target.fc_data.fPathPitchAngle
            # 目标马赫数
            tg_ma = target.fc_data.fMachNumber

        obs_dict["continuous"]["missile_attack_zone"] = {"dist": self.dist, "los_yaw_body": self.los_yaw_body,
                                                         "los_pitch_body": self.los_pitch_body, "AA_hori": AA_hori,
                                                         "pitch_self": fighter.fc_data.fPitchAngle,  # 自身俯仰角
                                                         "target_v_pitch": target_v_pitch,  # 目标爬升角
                                                         "alt_self": fighter.fc_data.fAltitude,  # 自身高度
                                                         "ma_self": fighter.fc_data.fMachNumber,  # 自身马赫数
                                                         "ma_target": tg_ma}  # 目标马赫数
        obs_dict["continuous"]["others"] = {"survive_info_self": fighter.combat_data.survive_info,  # 存活为True
                                            "v_pitch_self": fighter.fc_data.fPathPitchAngle,  # 自身爬升角
                                            "alpha_self": fighter.fc_data.fAttackAngle,  # 自身攻角
                                            "beta_self": fighter.fc_data.fSideslipAngle,  # 自身侧滑角
                                            "pitch_rate_self": fighter.fc_data.fPitchRate,  # 自身俯仰速率
                                            "roll_self": fighter.fc_data.fRollAngle,  # 自身滚转角
                                            "roll_rate_self": fighter.fc_data.fRollRate,  # 自身滚转速率
                                            "yaw_rate_self": fighter.fc_data.fYawRate,  # 自身偏航速律
                                            "normal_load_self": fighter.fc_data.fNormalLoad,  # 自身法向过载
                                            "lateral_load_self": fighter.fc_data.fLateralLoad,  # 自身侧向过载
                                            "longitude_load_self": fighter.fc_data.fLongitudeinalLoad,  # 自身机体纵向过载
                                            "fuel_self": fighter.fc_data.fNumberofFuel,  # 自身油量
                                            "dist_dot": dist_dot, "ATA": ATA}

        # 假设每架战斗机最多携带4枚导弹，每枚导弹的状态分为：0 未发射；1 飞行中；2 命中；3 失效，如果总数不足8枚，则不足的部分全为失效状态
        if len(fighter.missiles) >= 4:
            missiles_self_state_obs = [fighter.missiles[m].state for m in range(4)]
        else:
            missiles_self_state_obs = [fighter.missiles[m].state for m in range(len(fighter.missiles))]
            missiles_self_state_obs += [3 for m in range(4 - len(fighter.missiles))]

        # 将导弹告警进行符号化表示，特殊符号为0，表示没有导弹告警，假设最多有4个导弹告警
        missile_alerts = [[0, 0] for _ in range(4)]
        for m_a in range(min(fighter.sensors.alert_missile, 4)):
            # 先归一化到【-1，1】
            missile_alerts[m_a] = [fighter.sensors.alert_pitch[m_a] / 90, fighter.sensors.alert_yaw[m_a] / 180]
            # 在离散化，0为特殊符号所以1为起点
            missile_alerts[m_a] = [discrete_alert(missile_alerts[m_a][0], 100),
                                   discrete_alert(missile_alerts[m_a][1], 100)]

        obs_dict["discrete"]["missiles_self_state"] = missiles_self_state_obs
        obs_dict["discrete"]["missile_alert_tokens"] = missile_alerts

        return obs_dict

    def normalized_observation(self, fighter, target, env):
        """
        :param fighter: 智能体对象
        :param target: 智能体观测目标
        :param env: WVRENV_PHD
        :return: 归一化后的obs字典：分连续观测和离散观测, 连续观测中区分导弹发射相关状态、和非导弹发射相关状态
        """
        obs_dict = self.partially_observe(fighter, target, env)
        for key, value in obs_dict["continuous"]["missile_attack_zone"].items():
            if np.isnan(value):
                obs_dict["continuous"]["missile_attack_zone"][key] = 0.
                continue

            if key == 'dist':
                obs_dict["continuous"]["missile_attack_zone"][key] = 2 * logistic(value, 0.00076, 0) - 1
            elif key == 'los_yaw_body':
                obs_dict["continuous"]["missile_attack_zone"][key] = value / 180
            elif key == 'los_pitch_body':
                obs_dict["continuous"]["missile_attack_zone"][key] = value / 90
            elif key == 'AA_hori':
                obs_dict["continuous"]["missile_attack_zone"][key] = value / 180
            elif key == 'pitch_self':
                obs_dict["continuous"]["missile_attack_zone"][key] = value / 90
            elif key == 'target_v_pitch':
                obs_dict["continuous"]["missile_attack_zone"][key] = value / 90
            elif key == 'alt_self':
                obs_dict["continuous"]["missile_attack_zone"][key] = 0.000105 * value if value < 2000 \
                    else logistic(value, 0.00053, 4500)
            elif key == 'ma_self':
                obs_dict["continuous"]["missile_attack_zone"][key] = logistic(float(value), 7, 0.75)
                # obs_dict["continuous"]["missile_attack_zone"][key] = value / 2.
            elif key == 'ma_target':
                obs_dict["continuous"]["missile_attack_zone"][key] = logistic(float(value), 7, 0.75)
                # obs_dict["continuous"]["missile_attack_zone"][key] = value / 2.

        for key, value in obs_dict["continuous"]["others"].items():
            if np.isnan(value):
                obs_dict["continuous"]["others"][key] = 0.
                continue

            if key == 'roll_self':
                obs_dict["continuous"]["others"][key] = value / 180
            elif key == 'roll_rate_self':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 0.018, 0) - 1
            elif key == 'normal_load_self':
                obs_dict["continuous"]["others"][key] = value / 9 if value >= 0 else value / 3
            elif key == 'lateral_load_self':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 1.1, 0) - 1
            elif key == 'fuel_self':
                obs_dict["continuous"]["others"][key] = value / 3200
            elif key == 'v_pitch_self':
                obs_dict["continuous"]["others"][key] = value / 90
            elif key == 'alpha_self':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 0.073, 0) - 1
            elif key == 'beta_self':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 0.056, 0) - 1
            elif key == 'pitch_rate_self':
                obs_dict["continuous"]["others"][key] = value / 25
            elif key == 'yaw_rate_self':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 0.058, 0) - 1
            elif key == 'longitude_load_self':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 0.56, 0) - 1
            elif key == 'dist_dot':
                obs_dict["continuous"]["others"][key] = 2 * logistic(value, 0.009, 0) - 1
            elif key == 'ATA':
                obs_dict["continuous"]["others"][key] = value / 180

        return obs_dict

    def reshape_observation(self, fighter, target, env):
        """

        :param fighter:
        :param target: 战斗机观测的目标
        :param env:
        :return: obs_dict {"obs_self":ndarray, "obs_target":ndarray, "mis_states":ndarray, "mis_alert":ndarray}
        """
        obs_dict_old = self.normalized_observation(fighter, target, env)

        obs_self = [
            obs_dict_old["continuous"]["missile_attack_zone"]["pitch_self"],
            obs_dict_old["continuous"]["others"]["roll_self"],
            obs_dict_old["continuous"]["missile_attack_zone"]["alt_self"],
            obs_dict_old["continuous"]["missile_attack_zone"]["ma_self"],
            obs_dict_old["continuous"]["others"]["v_pitch_self"],
            obs_dict_old["continuous"]["others"]["alpha_self"],
            obs_dict_old["continuous"]["others"]["beta_self"],
            obs_dict_old["continuous"]["others"]["roll_rate_self"],
            obs_dict_old["continuous"]["others"]["pitch_rate_self"],
            obs_dict_old["continuous"]["others"]["yaw_rate_self"],
            obs_dict_old["continuous"]["others"]["normal_load_self"],
            obs_dict_old["continuous"]["others"]["lateral_load_self"],
            obs_dict_old["continuous"]["others"]["longitude_load_self"],
            obs_dict_old["continuous"]["others"]["fuel_self"]
        ]

        obs_target = [
            obs_dict_old["continuous"]["missile_attack_zone"]["dist"],
            obs_dict_old["continuous"]["missile_attack_zone"]["los_yaw_body"],
            obs_dict_old["continuous"]["missile_attack_zone"]["los_pitch_body"],
            obs_dict_old["continuous"]["missile_attack_zone"]["AA_hori"],
            obs_dict_old["continuous"]["missile_attack_zone"]["target_v_pitch"],
            obs_dict_old["continuous"]["missile_attack_zone"]["ma_target"],
            obs_dict_old["continuous"]["others"]["dist_dot"],
            obs_dict_old["continuous"]["others"]["ATA"]
        ]

        obs_self = np.array(obs_self, dtype=np.float32) * obs_dict_old["continuous"]["others"]["survive_info_self"]
        obs_target = np.array(obs_target, dtype=np.float32)

        mis_states = np.array(obs_dict_old["discrete"]["missiles_self_state"], dtype=np.int32)
        mis_alerts = (np.array(obs_dict_old["discrete"]["missile_alert_tokens"], dtype=np.int32) *
                      obs_dict_old["continuous"]["others"]["survive_info_self"])

        obs_dict_new = {'survive': obs_dict_old["continuous"]["others"]["survive_info_self"],
                        'obs_self': obs_self, 'obs_target': obs_target,
                        'mis_states': mis_states, 'mis_alerts': mis_alerts}

        return obs_dict_new

    def rnd_input(self, fighter, target, env):
        obs_dict_old = self.normalized_observation(fighter, target, env)

        missile_attack = [
            obs_dict_old["continuous"]["missile_attack_zone"]["alt_self"],
            obs_dict_old["continuous"]["missile_attack_zone"]["ma_self"],
            obs_dict_old["continuous"]["missile_attack_zone"]["pitch_self"],
            obs_dict_old["continuous"]["missile_attack_zone"]["dist"],
            obs_dict_old["continuous"]["missile_attack_zone"]["los_yaw_body"],
            obs_dict_old["continuous"]["missile_attack_zone"]["los_pitch_body"],
            obs_dict_old["continuous"]["missile_attack_zone"]["ma_target"],
            obs_dict_old["continuous"]["missile_attack_zone"]["AA_hori"],
            obs_dict_old["continuous"]["missile_attack_zone"]["target_v_pitch"]
        ]

        missile_attack = np.array(missile_attack, dtype=np.float32)

        return {'survive': obs_dict_old["continuous"]["others"]["survive_info_self"],
                'mis_attack': missile_attack}
