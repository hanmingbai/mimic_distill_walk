from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # --- 加载动作 ---
        motion_files = [cfg.motion_file] if isinstance(cfg.motion_file, str) else cfg.motion_file
        self.motions = [MotionLoader(f, self.body_indexes, device=self.device) for f in motion_files]
        self.num_motions = len(self.motions)
        self.all_motion_lengths = torch.tensor([m.time_step_total for m in self.motions], device=self.device)

        # 打印动作信息
        print("-" * 50)
        print(f"[MotionCommand] Loaded {self.num_motions} motions.")
        for i in range(min(self.num_motions, 50)):
            print(f"  [{i:03d}] Path: {os.path.basename(motion_files[i])} | Length: {self.motions[i].time_step_total}")
        if self.num_motions > 50: print(f"  ... and {self.num_motions - 50} more.")
        print("-" * 50)

        # --- 向量化存储优化 ---
        self._setup_tensorized_storage()

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Bin 统计适配
        update_dt = env.cfg.decimation * env.cfg.sim.dt
        self.bin_counts = [int(m.time_step_total // (1 / update_dt)) + 1 for m in self.motions]
        self.max_bin_count = max(self.bin_counts)
        self.bin_failed_count = torch.zeros((self.num_motions, self.max_bin_count), dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros((self.num_motions, self.max_bin_count), dtype=torch.float, device=self.device)
        
        self.kernel = torch.tensor([self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device)
        self.kernel = self.kernel / self.kernel.sum()

        for key in ["error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
                    "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel",
                    "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin"]:
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    def _setup_tensorized_storage(self):
        """合并 Tensor，干掉 Python 循环"""
        max_len = self.all_motion_lengths.max().item()
        num_bodies = len(self.cfg.body_names)
        num_joints = self.motions[0].joint_pos.shape[-1]

        self.all_joint_pos = torch.zeros((self.num_motions, max_len, num_joints), device=self.device)
        self.all_joint_vel = torch.zeros((self.num_motions, max_len, num_joints), device=self.device)
        self.all_body_pos = torch.zeros((self.num_motions, max_len, num_bodies, 3), device=self.device)
        self.all_body_quat = torch.zeros((self.num_motions, max_len, num_bodies, 4), device=self.device)
        self.all_body_lin_vel = torch.zeros((self.num_motions, max_len, num_bodies, 3), device=self.device)
        self.all_body_ang_vel = torch.zeros((self.num_motions, max_len, num_bodies, 3), device=self.device)

        for i, m in enumerate(self.motions):
            l = m.time_step_total
            self.all_joint_pos[i, :l] = m.joint_pos
            self.all_joint_vel[i, :l] = m.joint_vel
            self.all_body_pos[i, :l] = m.body_pos_w
            self.all_body_quat[i, :l] = m.body_quat_w
            self.all_body_lin_vel[i, :l] = m.body_lin_vel_w
            self.all_body_ang_vel[i, :l] = m.body_ang_vel_w

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.all_joint_pos[self.motion_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.all_joint_vel[self.motion_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.all_body_pos[self.motion_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.all_body_quat[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.all_body_lin_vel[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.all_body_ang_vel[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.all_body_pos[self.motion_ids, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.all_body_quat[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.all_body_lin_vel[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.all_body_ang_vel[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor: return self.robot.data.joint_pos
    @property
    def robot_joint_vel(self) -> torch.Tensor: return self.robot.data.joint_vel
    @property
    def robot_body_pos_w(self) -> torch.Tensor: return self.robot.data.body_pos_w[:, self.body_indexes]
    @property
    def robot_body_quat_w(self) -> torch.Tensor: return self.robot.data.body_quat_w[:, self.body_indexes]
    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor: return self.robot.data.body_lin_vel_w[:, self.body_indexes]
    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor: return self.robot.data.body_ang_vel_w[:, self.body_indexes]
    @property
    def robot_anchor_pos_w(self) -> torch.Tensor: return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_quat_w(self) -> torch.Tensor: return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor: return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor: return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_env_ids = env_ids[episode_failed]
            m_ids = self.motion_ids[failed_env_ids]
            t_steps = self.time_steps[failed_env_ids]
            m_totals = self.all_motion_lengths[m_ids]
            m_bins = torch.tensor([self.bin_counts[i] for i in m_ids], device=self.device)
            
            # 修复：避开 clamp(Tensor, int, Tensor) 冲突
            raw_indices = (t_steps * m_bins) // torch.clamp(m_totals, min=1)
            bin_indices = torch.max(torch.zeros_like(raw_indices), torch.min(raw_indices, m_bins - 1))
            
            for i, env_idx in enumerate(failed_env_ids):
                self._current_bin_failed[m_ids[i], bin_indices[i]] += 1

        self.motion_ids[env_ids] = torch.randint(0, self.num_motions, (len(env_ids),), device=self.device)
        unique_m_ids = torch.unique(self.motion_ids[env_ids])
        for m_id in unique_m_ids:
            m_env_mask = (self.motion_ids[env_ids] == m_id)
            target_env_ids = env_ids[m_env_mask]
            m_bin_count = self.bin_counts[m_id]
            probs = self.bin_failed_count[m_id, :m_bin_count] + self.cfg.adaptive_uniform_ratio / float(m_bin_count)
            probs = torch.nn.functional.pad(probs.unsqueeze(0).unsqueeze(0), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate")
            probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
            probs /= probs.sum()
            sampled_bins = torch.multinomial(probs, len(target_env_ids), replacement=True)
            m_total = self.all_motion_lengths[m_id]
            self.time_steps[target_env_ids] = ((sampled_bins + sample_uniform(0.0, 1.0, (len(target_env_ids),), device=self.device)) / m_bin_count * (m_total - 1)).long()
            
            # 修复：熵计算兼容 Tensor
            self.metrics["sampling_entropy"][target_env_ids] = -(probs * (probs + 1e-12).log()).sum() / torch.log(torch.tensor(max(m_bin_count, 2), device=self.device))
            pmax, imax = probs.max(dim=0)
            self.metrics["sampling_top1_prob"][target_env_ids] = pmax
            self.metrics["sampling_top1_bin"][target_env_ids] = imax.float() / m_bin_count
    
    def _motion_weight_adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_env_ids = env_ids[episode_failed]
            m_ids = self.motion_ids[failed_env_ids]
            t_steps = self.time_steps[failed_env_ids]
            m_totals = self.all_motion_lengths[m_ids]
            m_bins = torch.tensor([self.bin_counts[i] for i in m_ids], device=self.device)
            
            # 修复：避开 clamp(Tensor, int, Tensor) 冲突
            raw_indices = (t_steps * m_bins) // torch.clamp(m_totals, min=1)
            bin_indices = torch.max(torch.zeros_like(raw_indices), torch.min(raw_indices, m_bins - 1))
            
            for i, env_idx in enumerate(failed_env_ids):
                self._current_bin_failed[m_ids[i], bin_indices[i]] += 1

        # --- 核心修改：按动作长度采样 ---
        # 权重设为动作的总帧数，长度越长，被选中的概率越大
        motion_weights = self.all_motion_lengths.float()
        self.motion_ids[env_ids] = torch.multinomial(motion_weights, len(env_ids), replacement=True)
        # ----------------------------

        unique_m_ids = torch.unique(self.motion_ids[env_ids])
        for m_id in unique_m_ids:
            m_env_mask = (self.motion_ids[env_ids] == m_id)
            target_env_ids = env_ids[m_env_mask]
            m_bin_count = self.bin_counts[m_id]
            probs = self.bin_failed_count[m_id, :m_bin_count] + self.cfg.adaptive_uniform_ratio / float(m_bin_count)
            probs = torch.nn.functional.pad(probs.unsqueeze(0).unsqueeze(0), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate")
            probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
            probs /= probs.sum()
            sampled_bins = torch.multinomial(probs, len(target_env_ids), replacement=True)
            m_total = self.all_motion_lengths[m_id]
            self.time_steps[target_env_ids] = ((sampled_bins + sample_uniform(0.0, 1.0, (len(target_env_ids),), device=self.device)) / m_bin_count * (m_total - 1)).long()
            
            # 修复：熵计算兼容 Tensor
            self.metrics["sampling_entropy"][target_env_ids] = -(probs * (probs + 1e-12).log()).sum() / torch.log(torch.tensor(max(m_bin_count, 2), device=self.device))
            pmax, imax = probs.max(dim=0)
            self.metrics["sampling_top1_prob"][target_env_ids] = pmax
            self.metrics["sampling_top1_bin"][target_env_ids] = imax.float() / m_bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0: return
        self._motion_weight_adaptive_sampling(env_ids)
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]], device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        root_ori[env_ids] = quat_mul(quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]), root_ori[env_ids])
        
        v_ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]], device=self.device)
        v_samples = sample_uniform(v_ranges[:, 0], v_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += v_samples[:, :3]
        root_ang_vel[env_ids] += v_samples[:, 3:]

        j_pos = self.joint_pos.clone()
        j_pos += sample_uniform(*self.cfg.joint_position_range, j_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        j_pos[env_ids] = torch.clip(j_pos[env_ids], limits[:, :, 0], limits[:, :, 1])
        
        self.robot.write_joint_state_to_sim(j_pos[env_ids], self.joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1), env_ids=env_ids)

    def _update_command(self):
        self.time_steps += 1
        done_mask = self.time_steps >= self.all_motion_lengths[self.motion_ids]
        done_env_ids = torch.where(done_mask)[0]
        if len(done_env_ids) > 0: self._resample_command(done_env_ids)

        r_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        r_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        m_anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        m_anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = r_anchor_pos.clone()
        delta_pos_w[..., 2] = m_anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(r_anchor_quat, quat_inv(m_anchor_quat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - m_anchor_pos)

        self.bin_failed_count = self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor"))
                self.goal_anchor_visualizer = VisualizationMarkers(self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor"))
                self.current_body_visualizers = [VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + n)) for n in self.cfg.body_names]
                self.goal_body_visualizers = [VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + n)) for n in self.cfg.body_names]
            self.current_anchor_visualizer.set_visibility(True); self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)): self.current_body_visualizers[i].set_visibility(True); self.goal_body_visualizers[i].set_visibility(True)
        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False); self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)): self.current_body_visualizers[i].set_visibility(False); self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized: return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = MISSING
    motion_file: str | list[str] = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    bin_count: int = 100 
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # --- 修改这里的可视化尺寸 ---
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_anchor"
    )
    # 默认为 (1.0, 1.0, 1.0)，改为 0.1 甚至 0.05
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2) 
    
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_body"
    )
    # body 标记通常可以比 anchor 更小一点，方便区分
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)