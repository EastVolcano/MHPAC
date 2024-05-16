import numpy as np
from scipy.spatial.transform import Rotation as R
from WVRENV_PHD.utils.GNCData import wgs84ToNED, ned_to_body, euler2vector, angle_2vec, ned_to_wgs84, vector_angle
import torch
import torch.nn as nn
from fighter_bot import VecDotCtrl

ati_btt_guide = VecDotCtrl()


def logistic(x, growth_rate, mid_point):
    if (-growth_rate * (x - mid_point)) > 700:
        S = 0
    else:
        S = 1 / (1 + np.exp(-growth_rate * (x - mid_point)))
    return S


def reward_func(mis_model, fighter, obs, next_obs, rnd_next_obs, target_obs, target_next_obs,
                target_rnd_next_obs, target, env):
    """
    基于导弹命中概率设计的奖励
    :param target_rnd_next_obs:
    :param mis_model: 导弹命中概率拟合网络
    :param target_next_obs:
    :param target_obs:
    :param rnd_next_obs: 更新后，为RND模型生成的观测字典
    :param next_obs: 更新后的观测
    :param obs: 未更新的观测
    :param target: fighter的导弹攻击目标
    :param fighter: reward对应的战斗机对象
    :param env: 环境
    :return:
    """
    mis_model.eval()

    r_e = 0
    r_s = 0

    be_captured = 0
    for t_mis in target.missiles:
        if (fighter.index in t_mis.target_list) and (t_mis.state == 0):
            be_captured = 1
            break

    missile_capture = False
    for mis in fighter.missiles:
        if (target.index in mis.target_list) and (mis.state == 0):
            missile_capture = True
            break

    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
    los = np.array([l_n, l_e, l_d])
    ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
    target_ati_vec = euler2vector(target.fc_data.fRollAngle, target.fc_data.fPitchAngle, target.fc_data.fYawAngle)
    ATA = vector_angle(ati_vec, los)
    AA = vector_angle(target_ati_vec, los)
    dist = np.linalg.norm(los)

    vel = np.array(
        [fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
    vel_t = np.array(
        [target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
    if dist > 1:
        dist_dot = np.linalg.norm(vel_t) * \
                   (vel_t.dot(los) / (np.linalg.norm(vel_t) * dist)) - \
                   np.linalg.norm(vel) * \
                   (vel.dot(los) / (np.linalg.norm(vel) * dist))
    else:
        dist_dot = 0

    r_rel_pos = 0
    # r_rel_pos = (ATA / 180 - 2) * logistic(AA / 180, 18, 0.5) - ATA / 180 + 1
    r_rel_pos = 0.2 * ((ATA / 180 - 2) * 0.5 * (-logistic(AA / 180, 18, 1 / 6) + logistic(AA / 180, 18, 5 / 6))
                       - ATA / 180 - 1)
    # 瞄准奖励
    los_vec_body = ned_to_body(los, fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                               fighter.fc_data.fRollAngle)
    los_pitch_body = np.rad2deg(-np.arctan2(los_vec_body[2], np.linalg.norm(los_vec_body[0:2])))
    los_yaw_body = np.rad2deg(np.arctan2(los_vec_body[1], los_vec_body[0]))
    # r_los_body = (1 - logistic(abs(los_yaw_body), 0.18, 30)) * (1 - logistic(abs(los_pitch_body), 0.19, 30))

    pitch_err, roll_err = ati_btt_guide.get_err(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                                                fighter.fc_data.fYawAngle, los)
    if fighter.sensors.alert_missile > 0:
        r_los_body = 0
    else:
        r_los_body = - 0.9 * (
                0.2 * ((los_yaw_body / 180) ** 2) + 1.2 * ((pitch_err / 180) ** 2) + 0.2 * ((roll_err / 180) ** 2))

    # 接近率奖励
    r_closure = 1.5 * (dist_dot / 500) * (1 - logistic(AA / 180, 18, 0.5)) * logistic(dist, 0.0029, 3500)
    r_closure = 0
    r_closure = logistic(dist_dot, 0.016, -50) * (- logistic(dist, 0.0029, 7000))
    # 防撞惩罚
    r_too_close = 0
    r_too_close = -1.5 * logistic(AA / 180, 18, 0.5) * (1 - logistic(dist, 0.02, 270))
    # 持续快滚惩罚
    r_too_roll = 0
    # r_too_roll = 0. if fighter.fc_data.fRollRate < 50 else - 0.003 * fighter.fc_data.fRollRate
    # 低速、大攻角惩罚
    # r_ma_alpha = 0
    r_ma_alpha = 1. * ((2 * logistic(fighter.fc_data.fMachNumber, 12., 0.3) - 2) *
                       logistic(fighter.fc_data.fAttackAngle, 0.86, 28))
    # 高度惩罚
    r_alt = (-4 * (1 - logistic(fighter.fc_data.fAltitude, 0.008, 1000)) *
             logistic(fighter.fc_data.fVerticalVelocity, 0.18, 0))
    # 被锁定惩罚
    r_be_captured = 0
    if be_captured:
        t_missile_attack_tensor = torch.as_tensor(target_rnd_next_obs['mis_attack'], dtype=torch.float32)
        with torch.no_grad():
            r_be_captured = torch.softmax(mis_model(t_missile_attack_tensor), dim=-1)[0]
            r_be_captured = - 0.2 * r_be_captured.item()
    # r_be_captured = -0.3 * be_captured
    # 导弹告警惩罚
    r_mis_alert = 0
    if fighter.sensors.alert_missile > 0:
        for alert_id in range(fighter.sensors.alert_missile):
            alert_body_vec = euler2vector(0,
                                          fighter.sensors.alert_pitch[alert_id], fighter.sensors.alert_yaw[alert_id])
            # 将告警方向转到NED系下
            r_ned2body = R.from_euler('ZYX', [fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                                              fighter.fc_data.fRollAngle], degrees=True).inv()
            R_Body2NED = r_ned2body.inv().as_matrix()
            alert_NED_vec = np.matmul(R_Body2NED, alert_body_vec)
            ppg_vec = -alert_NED_vec
            evade_ata = vector_angle(ati_vec, ppg_vec)
            r_mis_alert -= ((evade_ata / 180) ** 2)

    # 坠毁惩罚 坠地 失速和碰撞
    if (not next_obs['survive']) and (obs['survive']) and (not fighter.combat_data.be_effective_killed):
        r_e -= 200

    # 被导弹命中惩罚
    t_missile_state_array = target_obs['mis_states']
    t_missile_state_array_next = target_next_obs['mis_states']
    if (len(t_missile_state_array_next[np.where(t_missile_state_array_next == 2)])
            > len(t_missile_state_array[np.where(t_missile_state_array == 2)])):
        r_e -= 100
    # 发射导弹惩罚（之前发射过导弹，且这个导弹还在飞行中）
    missile_state_array = obs['mis_states']
    missile_state_array_next = next_obs['mis_states']
    if (len(missile_state_array_next[np.where(missile_state_array_next > 0)])
        > len(missile_state_array[np.where(missile_state_array > 0)])) and \
            (len(missile_state_array_next[np.where(missile_state_array_next == 1)]) > 1):
        r_e -= 30
    # 导弹失效惩罚
    left_missiles = min(len(missile_state_array_next[np.where(missile_state_array_next == 0)]), fighter.missiles_max)
    if (len(missile_state_array_next[np.where(missile_state_array_next == 3)])
            > len(missile_state_array[np.where(missile_state_array == 3)])):
        if left_missiles > 0:
            r_e -= 30
        else:
            r_e -= 50
    # 导弹命中奖励 (同时与导弹当前生命值有关)
    if (len(missile_state_array_next[np.where(missile_state_array_next == 2)])
            > len(missile_state_array[np.where(missile_state_array == 2)])):
        if fighter.combat_data.bloods > target.combat_data.bloods:
            r_e += 100
        else:
            r_e += 30

    # 导弹命中概率奖励
    # 如果目标未被导弹导引头截获 命中概率奖励应为0
    r_missile_attack_zone = 0
    if missile_capture:
        missile_attack_tensor = torch.as_tensor(rnd_next_obs['mis_attack'], dtype=torch.float32)
        with torch.no_grad():
            r_missile_attack_zone = torch.softmax(mis_model(missile_attack_tensor), dim=-1)[0]
            r_missile_attack_zone = 0.2 * r_missile_attack_zone.item()

    reward = (r_e + 0.5 * fighter.combat_data.survive_info *
              (r_rel_pos + r_mis_alert + r_missile_attack_zone + r_alt + r_ma_alpha + r_too_roll +
               r_too_close + r_closure + r_los_body + r_be_captured))

    return reward


def reward_baseline_1(fighter, obs, next_obs, target, env):
    """
    纯角度态势奖励函数 + 导弹命中、失效
    :param target: 导引头的目标
    :param next_obs:
    :param obs:
    :param fighter:
    :param env:
    :return:
    """

    l_n, l_e, l_d = wgs84ToNED(target.fc_data.fLatitude, target.fc_data.fLongitude, target.fc_data.fAltitude,
                               fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
    los = np.array([l_n, l_e, l_d])
    ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle)
    target_ati_vec = euler2vector(target.fc_data.fRollAngle, target.fc_data.fPitchAngle, target.fc_data.fYawAngle)
    ATA = vector_angle(ati_vec, los)
    AA = vector_angle(target_ati_vec, los)
    dist = np.linalg.norm(los)
    r_s = np.exp(-abs(dist - 5000) / (180 * 0.1)) * (0.5 * ((1 - AA / 180) + (1 - ATA / 180)))

    r_e = 0
    # 被导弹命中惩罚
    if fighter.combat_data.be_effective_killed and (obs['survive'] and (not next_obs['survive'])):
        r_e -= -100
    # 发射导弹的惩罚
    missile_state_array = obs['mis_states']
    missile_state_array_next = next_obs['mis_states']
    if (len(missile_state_array_next[np.where(missile_state_array_next == 0)])
            > len(missile_state_array[np.where(missile_state_array == 0)])):
        r_e -= 5
    # 导弹命中奖励
    if (len(missile_state_array_next[np.where(missile_state_array_next == 2)])
            > len(missile_state_array[np.where(missile_state_array == 2)])):
        r_e += 100 / fighter.missiles_max

    return r_e + fighter.combat_data.survive_info * r_s


def reward_baseline_2(fighter, obs, next_obs, target, env):
    """
    纯角度态势奖励函数 + 导弹命中、失效
    :param target: 导引头的目标
    :param next_obs:
    :param obs:
    :param fighter:
    :param env:
    :return:
    """
    r_e = 0
    # 被导弹命中惩罚
    if fighter.combat_data.be_effective_killed and (obs['survive'] and (not next_obs['survive'])):
        r_e -= -100
    # 发射导弹的惩罚
    missile_state_array = obs['mis_states']
    missile_state_array_next = next_obs['mis_states']
    if (len(missile_state_array_next[np.where(missile_state_array_next == 0)])
            > len(missile_state_array[np.where(missile_state_array == 0)])):
        r_e -= 5
    # 导弹命中奖励
    if (len(missile_state_array_next[np.where(missile_state_array_next == 2)])
            > len(missile_state_array[np.where(missile_state_array == 2)])):
        r_e += 100 / fighter.missiles_max

    reward = r_e
    return reward
