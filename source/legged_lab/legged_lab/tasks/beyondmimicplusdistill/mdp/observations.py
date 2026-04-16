from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply_inverse, yaw_quat

from legged_lab.tasks.beyondmimicplusdistill.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    ) # 获得姿态的error
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    ) # 获得位置的error

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, # robot's
        command.robot_anchor_quat_w, # robot's
        command.anchor_pos_w, # motion's
        command.anchor_quat_w, # motion's
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def motion_velocity_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(command_name)

    linvel = command.anchor_lin_vel_w # 世界系下的线速度
    angvel = command.anchor_ang_vel_w # 世界系下的角速度
    anchor_quat = command.anchor_quat_w
    # 将线速度和角速度从世界系转换到局部坐标系下
    linvel_local = quat_apply_inverse(anchor_quat, linvel)
    angvel_local = quat_apply_inverse(anchor_quat, angvel)
    # # 只考虑xy的投影，更适合z轴姿态变化较大的运动
    # quat_yaw = yaw_quat(anchor_quat)
    # linvel_local = quat_apply_inverse(quat_yaw, linvel)
    # angvel_local = quat_apply_inverse(quat_yaw, angvel)

    # linvel = torch.zeros_like(command.anchor_lin_vel_w)
    # angvel = torch.zeros_like(command.anchor_ang_vel_w)
    # linvel[:, 0] = 0.6
    # angvel[:, 2] = 0.5

    # print(f"command.anchor_lin_vel_w:{command.anchor_lin_vel_w.shape}")
    return torch.cat([linvel_local[:, :2], angvel_local[:, 2:3]], dim=-1)

# for perceptive distillation
def get_depth_data(env, sensor_name: str):
    """从指定的 TiledCamera 获取归一化的拉平深度数据"""
    # 拿到的是 [num_envs, 58, 87] 的深度 Tensor
    depth = env.scene[sensor_name].data.output["distance_to_image_plane"]
    # 预处理：NaN/Inf -> 5.0m, 并归一化到 [0, 1]
    depth = torch.nan_to_num(depth, posinf=5.0)
    depth = torch.clamp(depth / 5.0, 0.0, 1.0)
    # 拉平以便 ObservationManager 拼接成 1D Vector (5046维)
    return depth.view(env.num_envs, -1)
