import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from WVRENV.utils.GNCData import euler2vector, euler2quat, vector_angle, wgs84ToNED

K_W = 4  # 滚转速率P参数
K_W_D = 2  # 滚转速率D参数
K_N = 2.8  # 法向过载P参数
K_R = 1.5  # 俯仰抑制系数导数
K_Y = 0.07  # 方向舵参数
K_T = 2.5  # 油门参数


class VecDotCtrl(object):
    def __init__(self):
        # 当前机体向量
        self.ati_vec_cur = np.ones(3) / np.sqrt(3)
        # 期望机体向量
        self.ati_vec_des = np.ones(3) / np.sqrt(3)
        # 当前姿态四元数及逆
        self.quat_vec_cur_inv = np.ones(4) / 2
        self.quat_vec_cur = np.ones(4) / 2
        # 期望姿态四元数及逆
        self.quat_vec_des_inv = np.ones(4) / 2
        self.quat_vec_des = np.ones(4) / 2
        # 法向过载角误差 度
        self.theta_err = 0
        # 上一时刻的滚转误差
        self.last_roll_err = 0

    def vec_quat_cur(self, roll, pitch, yaw):
        rotation_1 = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
        self.quat_vec_cur_inv = rotation_1.as_quat()  # (x y z w) : (q1 q2 q3 q0)
        self.quat_vec_cur = R.from_quat(self.quat_vec_cur_inv).inv().as_quat()
        self.ati_vec_cur = euler2vector(roll, pitch, yaw)
        self.ati_vec_cur = self.ati_vec_cur / np.linalg.norm(self.ati_vec_cur)

    # print("quat cur: ", self.quat_vec_cur_inv)

    def vec_quat_des(self, vec_des):
        """
        获得期望向量+滚转的四元数
        :param vec_des: NED系下的期望向量
        :return:
        """
        if np.linalg.norm(vec_des) < 0.01:
            self.ati_vec_des[:] = 0
            return
        else:
            self.ati_vec_des[:] = vec_des / np.linalg.norm(vec_des)
        # 第一次旋转：俯仰，偏航
        pitch_d = np.arctan2(-vec_des[2], np.linalg.norm(vec_des[0:2]))
        yaw_d = np.arctan2(vec_des[1], vec_des[0])
        quat_des_1_inv = euler2quat(0, pitch_d, yaw_d)  # 第一次旋转的逆
        # print("quat_des_1_inv: ", quat_des_1_inv)

        # 第二次旋转：加上滚转
        r_axis_1 = np.cross(self.ati_vec_cur, self.ati_vec_des)  # 当前向量转到期望向量的转轴
        if np.linalg.norm(r_axis_1) > 0.001:
            ey_d = r_axis_1 / np.linalg.norm(r_axis_1)  # 期望机体y轴在NED的单位向量
        else:
            ey_d = R.from_quat(quat_des_1_inv).as_matrix()[:, 1]

        self.theta_err = vector_angle(self.ati_vec_cur, self.ati_vec_des)  # 当前向量与期望向量的空间角误差

        ey_1 = R.from_quat(quat_des_1_inv).as_matrix()[:, 1]  # 第一次旋转后的机体y轴单位向量

        r_axis_2 = np.cross(ey_1, ey_d)
        if np.linalg.norm(r_axis_2) > 0.01:
            r_axis_2 = r_axis_2 / np.linalg.norm(r_axis_2)  # 第二次旋转的转轴, 但实际转轴为期望机头向量
        angel_r2 = vector_angle(ey_1, ey_d) * np.sign(np.dot(r_axis_2, self.ati_vec_des))

        # print("roll err 1:", angel_r2)

        # print("r2: ", r_axis_2, "angel_r2: ", np.deg2rad(angel_r2))
        # print("quat_des_2: ", quat_des_2)

        # 复合两次旋转 ： (R2R1)^T = R1^T * R2^T
        rotate_x_2 = R.from_euler('X', [-angel_r2], degrees=True)
        rotate_x_2_inv = rotate_x_2.inv()
        rotate_1_inv = R.from_quat(quat_des_1_inv)
        self.quat_vec_des_inv = R.from_matrix(np.matmul(rotate_1_inv.as_matrix(), rotate_x_2_inv.as_matrix())).as_quat()
        self.quat_vec_des_inv = np.squeeze(self.quat_vec_des_inv)

    def get_cmd(self):
        # 当前与期望 机体y轴的单位向量（NED）
        ey_cur = R.from_quat(self.quat_vec_cur_inv).as_matrix()[:, 1]
        ey_des = R.from_quat(self.quat_vec_des_inv).as_matrix()[:, 1]

        # 分离滚转四元数
        rx = np.cross(ey_cur, ey_des)  # 滚转转轴，但实际转轴是当前机头向量
        if np.linalg.norm(rx) > 0.01:
            rx = rx / np.linalg.norm(rx)
        roll_angle = vector_angle(ey_cur, ey_des) * np.sign(np.around(100000000 * np.dot(rx, self.ati_vec_cur)))  # 滚转转角

        # print("roll err2: ", roll_angle)
        # omega = K_W * (roll_angle) + K_W_D * (0 - wx)
        # 在滚转误差较大时，抑制俯仰（法向过载）
        # 抑制比例
        # k_scale = (180 - K_R * abs(roll_angle)) / 180
        # print("test: ", roll_angle)
        # 当俯仰误差较小，滚转误差接近180，利用负过载改变机头指向
        pitch_angle = self.theta_err

        # load = k_scale * K_N * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8 # 可考虑加入导数项
        # load = k_scale * K_N * vel * pitch_angle / 9.8
        # 考虑加重补？
        return pitch_angle, roll_angle

    def get_err(self, roll, pitch, yaw, vec_des):
        """

        :param roll: 战斗机滚转 deg
        :param pitch: 战斗机俯仰 deg
        :param yaw: 战斗机偏航 deg
        :param vec_des: 期望机头指向
        :return: 法向过载、滚转速率
        """
        self.vec_quat_cur(roll, pitch, yaw)
        self.vec_quat_des(vec_des)
        pitch_angle, roll_angle = self.get_cmd()
        return pitch_angle, roll_angle

    def ctrl_cmd(self, Ma, vel, roll, pitch, yaw, vec_des, dt):
        """

        :param Ma: 马赫数
        :param vel: 真空速
        :param roll: 战斗机滚转 deg
        :param pitch: 战斗机俯仰 deg
        :param yaw: 战斗机偏航 deg
        :param vec_des: 期望机头指向
        :param dt: 调用函数的时间步长(仿真一步的时间为0.01s)
        :return: 法向过载、滚转速率
        """
        self.vec_quat_cur(roll, pitch, yaw)
        self.vec_quat_des(vec_des)
        pitch_angle, roll_angle = self.get_cmd()
        # print(f"roll dot: {roll_vec_dot}, pitch dot: {pitch_vec_dot}")
        omega = K_W * (roll_angle) + K_W_D * ((roll_angle - self.last_roll_err) / dt)
        # 滚转死区
        if (pitch_angle < 2) and (roll_angle > 30):
            omega = 0
        # print(f"roll angle: {roll_angle}, roll_dot: {np.rad2deg(roll_vec_dot)}, "
        # 	  f"pitch angel: {pitch_angle}, pitch_dot: {np.rad2deg(pitch_vec_dot)}")
        # 抑制比例
        k_scale = (180 - K_R * abs(roll_angle)) / 180
        load = K_N * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8
        if (Ma > 0.95):
            load = K_N * 0.6 * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8
        elif (Ma > 1.4):
            load = K_N * 0.2 * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8
        load *= k_scale

        # 偏航
        coordinate_trans = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).inv()
        trd_body_los = np.matmul(coordinate_trans.as_matrix(), vec_des)
        # print(f"body los: {trd_body_los}")
        body_yaw_des = np.rad2deg(np.arctan2(trd_body_los[1], trd_body_los[0]))
        body_pitch_des = np.rad2deg(-np.arctan2(trd_body_los[2], np.linalg.norm(trd_body_los[0:2])))
        if (pitch_angle > 120) and (body_yaw_des > 90) and (roll_angle > 40):
            load = 1
        # print(f"body_yaw_des: {body_yaw_des}")
        rudder = K_Y * np.deg2rad(body_yaw_des)
        if abs(rudder) > 1:
            rudder = np.sign(rudder)
        # 油门
        thrust = 0.5 + K_T * (0.9 - Ma)
        if thrust < 0.05:
            thrust = 0.05
        elif thrust > 1:
            thrust = 1
        self.last_roll_err = roll_angle
        return thrust, load, omega, rudder

    def trainable_cmd(self, Ma, vel, roll, pitch, yaw, vec_des, dt, target_vel_array,
                      k_n=2.6, k_n_b=0, k_r=1.5, k_w=4, k_w_d=2, k_y=0.07, k_t=2.5):
        """

        :param Ma: 马赫数
        :param vel: 真空速
        :param roll: 战斗机滚转 deg
        :param pitch: 战斗机俯仰 deg
        :param yaw: 战斗机偏航 deg
        :param vec_des: 期望机头指向
        :param dt: 调用函数的时间步长(仿真一步的时间为0.01s)
        :param target_vel_array: 目标速度矢量
        :param k_n:
        :param k_n_b:
        :param k_r:
        :param k_w:
        :param k_w_d:
        :param k_y:
        :param k_t:
        :return: 法向过载、滚转速率
        """
        self.vec_quat_cur(roll, pitch, yaw)
        self.vec_quat_des(vec_des)
        pitch_angle, roll_angle = self.get_cmd()

        coordinate_trans = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).inv()
        # 目标视线矢量在机体系下的表示
        trd_body_los = np.matmul(coordinate_trans.as_matrix(), vec_des)
        # 目标速度矢量在机体系下的表示
        body_tgt_vel = np.matmul(coordinate_trans.as_matrix(), target_vel_array)
        # print(f"body los: {trd_body_los}")
        body_yaw_des = np.rad2deg(np.arctan2(trd_body_los[1], trd_body_los[0]))
        body_pitch_des = np.rad2deg(-np.arctan2(trd_body_los[2], np.linalg.norm(trd_body_los[0:2])))
        # 本机俯仰平面内视线与目标速度的夹角
        body_tgt_pitch = np.rad2deg(np.arctan2(body_tgt_vel[1], body_tgt_vel[0]))

        if (abs(body_yaw_des) < 30) and (abs(roll_angle) < 30):
            pitch_angle_b = body_tgt_pitch - body_yaw_des
        else:
            pitch_angle_b = 0

        # 控制指令计算
        omega = k_w * (roll_angle) + k_w_d * ((roll_angle - self.last_roll_err) / dt)
        # 滚转死区
        if (pitch_angle < 2) and (roll_angle > 30):
            omega = 0
        # print(f"roll angle: {roll_angle}, roll_dot: {np.rad2deg(roll_vec_dot)}, "
        # 	  f"pitch angel: {pitch_angle}, pitch_dot: {np.rad2deg(pitch_vec_dot)}")
        # 抑制比例
        k_scale = (180 - k_r * abs(roll_angle)) / 180
        load = k_n * vel * 2 * np.sin(np.deg2rad(pitch_angle) / 2) / 9.8 + \
               k_n_b * vel * 2 * np.sin(np.deg2rad(pitch_angle_b) / 2) / 9.8
        load *= k_scale
        # print(f"body_yaw_des: {body_yaw_des}")
        rudder = k_y * np.deg2rad(body_yaw_des)
        if abs(rudder) > 1:
            rudder = np.sign(rudder)
        # 油门
        thrust = 0.5 + k_t * (0.9 - Ma)
        if thrust < 0.1:
            thrust = 0.1
        elif thrust > 1:
            thrust = 1
        self.last_roll_err = roll_angle
        return thrust, load, omega, rudder


class ManeuversLib(object):
    def __init__(self, decision_dt):
        # 机动动作被调用的步数
        self.decision_count = None
        # 机动动作编号
        self.maneuver_index = None
        # kar-PPG 动作是否为PPG的开关
        self.ppg_switch = True
        # 进入破S的开关
        self.spiltS_switch = False
        # 进入简单桶滚的开关
        self.simple_barrel_switch = False
        # 战斗机进入机动动作时的初始姿态（roll pitch yaw）
        self.ini_ati_angle = [0, 0, 0]
        # PPG 的实际跟踪方向（NED）
        self.ppg_vec = None
        # PPG 动作对象
        self.ppg = VecDotCtrl()  # 跟踪NED下的向量
        # 仿真调用机动动作的决策步长(仿真一步的时间为0.01s)
        self.decision_dt = decision_dt

    def __len__(self):
        """

        :return: 机动库的机动动作数
        """
        return 7

    def reset_lib(self, fighter, maneuver_index):
        """
        开始一次仿真前就要重置动作编号和步数
        :param fighter: 环境中的战斗机对象
        :param maneuver_index: 设置的机动动作编号
        :return:
        """
        self.maneuver_index = maneuver_index
        self.decision_count = 0
        self.ppg_vec = None
        self.ppg_switch = True
        self.spiltS_switch = False
        self.ini_ati_angle = [fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle]

    def step_decision(self, fighter, bogy):
        """
        每一个决策步调用一次机动动作
        :param bogy: 环境中的战斗机对象 (发射导弹的载机)
        :param fighter: 环境中的战斗机对象 (导弹的目标)
        :return: 油门、俯仰、滚转、偏航指令
        """
        thrust, load, omega, rudder = 1, 1, 0, 0
        if self.maneuver_index == 0:
            # 直线飞行
            thrust, load, omega, rudder = self.line_3d(fighter)
        elif self.maneuver_index == 1:
            # 简单桶滚
            thrust, load, omega, rudder = self.simple_barrel(fighter, fighter.sensors.alert_missile)
        elif self.maneuver_index == 2:
            # 破S
            thrust, load, omega, rudder = self.spilt_s(fighter, fighter.sensors.alert_missile)
        elif self.maneuver_index == 3:
            # 置尾
            if fighter.sensors.alert_missile > 0:
                thrust, load, omega, rudder = self.trail_missile(fighter, fighter.sensors.alert_pitch[0],
                                                                 fighter.sensors.alert_yaw[0])
            else:
                thrust, load, omega, rudder = self.trail_missile(fighter, None, None)
        elif self.maneuver_index == 4:
            # k = 1 ，kar = 0.2 的 kar-PPG
            l_n, l_e, l_d = wgs84ToNED(bogy.fc_data.fLatitude, bogy.fc_data.fLongitude, bogy.fc_data.fAltitude,
                                       fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
            thrust, load, omega, rudder = self.kar_ppg(fighter, kar=0.2, k=1, target_los=[l_n, l_e, l_d])
        elif self.maneuver_index == 5:
            # k = 2.8 ，kar = 0.1 的 kar-PPG
            l_n, l_e, l_d = wgs84ToNED(bogy.fc_data.fLatitude, bogy.fc_data.fLongitude, bogy.fc_data.fAltitude,
                                       fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
            thrust, load, omega, rudder = self.kar_ppg(fighter, kar=0.1, k=2.8, target_los=[l_n, l_e, l_d])
        elif self.maneuver_index == 6:
            # k = 2.8 ，kar = 0.7 的 kar-PPG
            l_n, l_e, l_d = wgs84ToNED(bogy.fc_data.fLatitude, bogy.fc_data.fLongitude, bogy.fc_data.fAltitude,
                                       fighter.fc_data.fLatitude, fighter.fc_data.fLongitude, fighter.fc_data.fAltitude)
            thrust, load, omega, rudder = self.kar_ppg(fighter, kar=0.7, k=2.8, target_los=[l_n, l_e, l_d])
        else:
            thrust, load, omega, rudder = self.line_3d(fighter)

        return thrust, load, omega, rudder

    def kar_ppg(self, fighter, kar=0., k=3., target_los=None):
        """
        随机转向一个方向 + 纯追踪制导律
        :param fighter: 环境中的战斗机对象
        :param kar: 随机方向的概率 0~1
        :param k: PPG的参数
        :param target_los: PPG追踪的方向
        :return: 油门、俯仰、滚转、偏航指令
        """

        # PPG
        if target_los is None:
            target_los = [1, 0, 0]

        if (self.decision_count * int(self.decision_dt / 0.01)) % 600 == 0:   # 每7秒进行一次判断，默认一步仿真时间为0.01s
            if np.random.uniform(0, 1, 1)[0] < kar:
                self.ppg_switch = False
                ati_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                                       fighter.fc_data.fYawAngle)
                random_yaw = np.random.uniform(-70, 70, 1)[0]
                random_pitch = np.random.uniform(-40, 40, 1)[0]
                rand_t = R.from_euler('ZYX', [random_yaw, random_pitch, 0], degrees=True)
                self.ppg_vec = np.matmul(rand_t.as_matrix(), ati_vec)
                # 防止倾角过大或过小
                des_path_picth = np.arctan2(-self.ppg_vec[2], np.linalg.norm(self.ppg_vec[0:2]))
                if abs(des_path_picth) > (np.pi / 3):
                    self.ppg_vec[2] = np.sign(self.ppg_vec[2]) * np.linalg.norm(self.ppg_vec[0:2]) * 1.7
            else:
                self.ppg_switch = True
        # print(f"debug time: {self.decision_count}, switch: {self.ppg_switch}, vec: {self.ppg_vec}")
        if self.ppg_switch:
            self.ppg_vec = np.array(target_los)
        # print("count: ", self.count, "guide_state: ", self.guide_state, "blue ati_vec_des: ", self.act_ati_vec )

        thrust, load, omega, rudder = self.ppg.trainable_cmd(fighter.fc_data.fMachNumber,
                                                             fighter.fc_data.fTrueAirSpeed,
                                                             fighter.fc_data.fRollAngle,
                                                             fighter.fc_data.fPitchAngle,
                                                             fighter.fc_data.fYawAngle, self.ppg_vec,
                                                             self.decision_dt, [100, 0, 0], k_n=k)
        self.decision_count += 1

        return thrust, load, omega, rudder

    def trail_missile(self, fighter, alert_body_pitch=None, alert_body_yaw=None):
        """
        置尾机动: 向导弹告警的反方向逃逸, 如果没有告警则延当前方向直线
        :param fighter: 环境中的战斗机对象
        :param alert_body_yaw: 导弹告警方向在机体系下的方位角 deg
        :param alert_body_pitch: 导弹告警方向在机体系的高低角 deg
        :return: 油门、俯仰、滚转、偏航指令
        """
        self.decision_count += 1

        if (alert_body_pitch is None) or (alert_body_yaw is None):
            alert_body_vec = [-1, 0, 0]  # 导弹告警方向矢量在机体系下投影
        else:
            alert_body_vec = euler2vector(0, alert_body_pitch, alert_body_yaw)

        # 将告警方向转到NED系下
        r_ned2body = R.from_euler('ZYX', [fighter.fc_data.fYawAngle, fighter.fc_data.fPitchAngle,
                                          fighter.fc_data.fRollAngle], degrees=True).inv()
        R_Body2NED = r_ned2body.inv().as_matrix()
        alert_NED_vec = np.matmul(R_Body2NED, alert_body_vec)

        # 用PPG向反方向跟踪
        self.ppg_vec = -alert_NED_vec
        thrust, load, omega, rudder = self.ppg.trainable_cmd(fighter.fc_data.fMachNumber,
                                                             fighter.fc_data.fTrueAirSpeed,
                                                             fighter.fc_data.fRollAngle,
                                                             fighter.fc_data.fPitchAngle,
                                                             fighter.fc_data.fYawAngle, self.ppg_vec,
                                                             self.decision_dt, [100, 0, 0])
        return thrust, load, omega, rudder

    def spilt_s(self, fighter, alert_num):
        """
        在收到导弹告警时做破S机动
        :param fighter: 环境中的战斗机对象
        :param alert_num: 导弹告警数量
        :return: 油门、俯仰、滚转、偏航指令
        """
        self.decision_count += 1

        if alert_num > 0:
            # 一旦有导弹告警后就永远进入破S机动
            self.spiltS_switch = True

        if self.spiltS_switch:
            if ((self.decision_count * int(self.decision_dt / 0.01)) < 1000) and (fighter.fc_data.fAltitude > 1000):
                # 10秒判断，默认一步仿真时间0.01s
                self.ppg_vec = np.array([0, 0, 1])
            else:
                if fighter.fc_data.fAltitude > 600:
                    self.ppg_vec = euler2vector(0, 0, self.ini_ati_angle[2] + 90)
                else:
                    self.ppg_vec = euler2vector(0, 5, self.ini_ati_angle[2] + 90)
        # 如果没有导弹告警，则直线飞行
        else:
            # 如果没有进入破S，则一直更新飞机进入破S的初始姿态
            self.ini_ati_angle = [fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle, fighter.fc_data.fYawAngle]
            self.ppg_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                                        fighter.fc_data.fYawAngle)

        thrust, load, omega, rudder = self.ppg.trainable_cmd(fighter.fc_data.fMachNumber,
                                                             fighter.fc_data.fTrueAirSpeed,
                                                             fighter.fc_data.fRollAngle,
                                                             fighter.fc_data.fPitchAngle,
                                                             fighter.fc_data.fYawAngle, self.ppg_vec,
                                                             self.decision_dt, [100, 0, 0])
        return thrust, load, omega, rudder

    def simple_barrel(self, fighter, alert_num):
        """
        在收到导弹告警时做简单桶滚机动
        :param fighter: 环境中的战斗机对象
        :param alert_num: 导弹告警数量
        :return: 油门、俯仰、滚转、偏航指令
        """
        self.decision_count += 1

        if alert_num > 0:
            # 有导弹告警后就永远进入简单桶滚机动
            self.simple_barrel_switch = True
        else:
            # 如果没有导弹告警，则退出简单桶滚机动
            self.simple_barrel_switch = False

        if self.simple_barrel_switch:
            load = 8
            omega = 80
            rudder = 0
            thrust = 1
        # 如果没有导弹告警，则直线飞行
        else:
            self.ppg_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                                        fighter.fc_data.fYawAngle)
            thrust, load, omega, rudder = self.ppg.trainable_cmd(fighter.fc_data.fMachNumber,
                                                                 fighter.fc_data.fTrueAirSpeed,
                                                                 fighter.fc_data.fRollAngle,
                                                                 fighter.fc_data.fPitchAngle,
                                                                 fighter.fc_data.fYawAngle, self.ppg_vec,
                                                                 self.decision_dt, [100, 0, 0])

        return thrust, load, omega, rudder

    def line_3d(self, fighter):
        """
        三维直线飞行
        :param fighter: 环境中的战斗机对象
        :return: 油门、俯仰、滚转、偏航指令
        """
        self.decision_count += 1

        self.ppg_vec = euler2vector(fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                                    fighter.fc_data.fYawAngle)
        thrust, load, omega, rudder = self.ppg.trainable_cmd(fighter.fc_data.fMachNumber,
                                                             fighter.fc_data.fTrueAirSpeed,
                                                             fighter.fc_data.fRollAngle,
                                                             fighter.fc_data.fPitchAngle,
                                                             fighter.fc_data.fYawAngle, self.ppg_vec,
                                                             self.decision_dt, [100, 0, 0])
        return thrust, load, omega, rudder
