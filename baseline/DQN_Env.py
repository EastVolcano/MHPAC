from baseline.warcraft_red import Warcraft, wgs84_2_ned, angle_error
import numpy as np


rad2deg = 1 / np.pi * 180
deg2rad = 1 / 180 * np.pi


def get_state(fighter, target):
    # state
    height = fighter.fc_data.fAltitude
    vel = np.array([fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity])
    vel_norm = (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5
    vel_target = np.array([target.fc_data.fNorthVelocity, target.fc_data.fEastVelocity, target.fc_data.fVerticalVelocity])
    vel_target_norm = (vel_target[0] ** 2 + vel_target[1] ** 2 + vel_target[2] ** 2) ** 0.5
    delta_pos = wgs84_2_ned(target.fc_data.fLatitude, target.fc_data.fLongitude,
                            target.fc_data.fAltitude,
                            fighter.fc_data.fLatitude, fighter.fc_data.fLongitude,
                            fighter.fc_data.fAltitude)
    d = (delta_pos[0] ** 2 + delta_pos[1] ** 2 + delta_pos[2] ** 2) ** 0.5
    # Heading Crossing Angle HCA 航向夹角：敌我双方速度矢量的夹角
    HCA = np.arccos(np.inner(vel, vel_target) / vel_norm / vel_target_norm)
    # Antenna train angle ATA 方位角：自身速度矢量与视线矢量的夹角
    ATA = np.arccos(np.inner(delta_pos, vel) / d / vel_norm)
    # Aspect Angle 敌方方位角：敌方速度矢量与视线矢量的夹角
    AA = np.arccos(np.inner(delta_pos, vel_target) / d / vel_target_norm)
    # Path Pitch Desired：预期俯仰角
    PPD = np.arctan2(- delta_pos[2], (delta_pos[0] ** 2 + delta_pos[1] ** 2) ** 0.5)
    # Path Yaw Desired：预期航向角
    PYD = np.arctan2(delta_pos[1], delta_pos[0])
    angle_relation = np.array([HCA, ATA, AA, PPD])

    angle_self = np.array([fighter.fc_data.fRollAngle, fighter.fc_data.fPitchAngle,
                           angle_error(PYD * rad2deg, fighter.fc_data.fYawAngle),
                           fighter.fc_data.fPathPitchAngle,
                           angle_error(PYD * rad2deg, fighter.fc_data.fPathYawAngle)])
    state = np.concatenate([np.array([height / 2000]),
                            angle_self / 180,
                            np.array([vel_norm / 200]),
                            np.array([d / 2000]),
                            angle_relation / np.pi,
                            np.array([vel_target_norm / 200])])
    return state
