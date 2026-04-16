"""
批量转换脚本：将文件夹下的所有 CSV 动作文件转换为 NPZ 格式
用法:
python scripts/csv_to_npz_batch.py --input_dir ./datasets/seed_g1 --output_dir ./motions --input_fps 120 --output_fps 50
"""

import argparse
import numpy as np
import os
import glob
import torch

from isaaclab.app import AppLauncher

# 1. 配置命令行参数
parser = argparse.ArgumentParser(description="Batch replay motions from a directory and output to npz files.")
parser.add_argument("--input_dir", type=str, help="包含 CSV 文件的文件夹路径")
parser.add_argument("--input_file", type=str, help="单个 CSV 文件路径（备选）")
parser.add_argument("--output_dir", type=str, default="./motions", help="输出 npz 的文件夹")
parser.add_argument("--input_fps", type=int, default=120, help="输入数据的 FPS")
parser.add_argument("--output_fps", type=int, default=50, help="输出数据的 FPS")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="帧范围: START END (从1开始)",
)

# 启动 Isaac Sim
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入 Isaac Lab 组件
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp, quat_from_euler_xyz

from legged_lab.robots.g1 import G1_CYLINDER_CFG

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """加载数据：厘米->米，欧拉角->四元数，角度->弧度，XY归零"""
        if self.frame_range is None:
            raw_data = np.loadtxt(self.motion_file, delimiter=",", skiprows=1)
        else:
            start_row = self.frame_range[0]
            num_rows = self.frame_range[1] - self.frame_range[0] + 1
            raw_data = np.loadtxt(self.motion_file, delimiter=",", skiprows=start_row, max_rows=num_rows)
        
        full_motion = torch.from_numpy(raw_data).to(torch.float32).to(self.device)
        
        # 1. 去掉第一列 Frame
        motion = full_motion[:, 1:]

        # 2. Root 位置 (cm -> m)
        self.motion_base_poss_input = motion[:, 0:3] * 0.01
        # XY 归零
        self.motion_base_poss_input[:, 0] -= self.motion_base_poss_input[0, 0]
        self.motion_base_poss_input[:, 1] -= self.motion_base_poss_input[0, 1]

        # 3. Root 姿态 (Euler deg -> Quat WXYZ)
        euler_rad = motion[:, 3:6] * (np.pi / 180.0)
        self.motion_base_rots_input = quat_from_euler_xyz(euler_rad[:, 0], euler_rad[:, 1], euler_rad[:, 2])

        # 4. 关节角度 (deg -> rad)
        self.motion_dof_poss_input = motion[:, 6:] * (np.pi / 180.0)

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self.motion_base_poss_input[index_0] * (1 - blend.unsqueeze(1)) + self.motion_base_poss_input[index_1] * blend.unsqueeze(1)
        
        slerped_rots = torch.zeros((self.output_frames, 4), device=self.device)
        for i in range(self.output_frames):
            slerped_rots[i] = quat_slerp(self.motion_base_rots_input[index_0[i]], self.motion_base_rots_input[index_1[i]], blend[i])
        self.motion_base_rots = slerped_rots
        
        self.motion_dof_poss = self.motion_dof_poss_input[index_0] * (1 - blend.unsqueeze(1)) + self.motion_dof_poss_input[index_1] * blend.unsqueeze(1)

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        
        rotations = self.motion_base_rots
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag: self.current_idx = 0
        return state, reset_flag

def run_simulator(sim, scene, joint_names, input_file, output_file):
    motion = MotionLoader(input_file, args_cli.input_fps, args_cli.output_fps, sim.device, args_cli.frame_range)
    robot = scene["robot"]
    robot_idx = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {"fps": [args_cli.output_fps], "joint_pos": [], "joint_vel": [], "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}
    
    while simulation_app.is_running():
        (m_pos, m_rot, m_lvel, m_avel, m_dpos, m_dvel), reset_flag = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = m_pos
        root_states[:, 3:7] = m_rot
        root_states[:, 7:10] = m_lvel
        root_states[:, 10:] = m_avel
        robot.write_root_state_to_sim(root_states)

        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        joint_pos[:, robot_idx], joint_vel[:, robot_idx] = m_dpos, m_dvel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        sim.render()
        scene.update(sim.get_physics_dt())

        # 记录数据
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if reset_flag:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            for k in log: 
                if k != "fps": log[k] = np.stack(log[k], axis=0)
            np.savez(output_file, **log)
            print(f"[SUCCESS]: Saved {output_file}")
            return

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    joint_list = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]

    files = glob.glob(os.path.join(args_cli.input_dir, "*.csv")) if args_cli.input_dir else [args_cli.input_file]
    for f in files:
        out = os.path.join(args_cli.output_dir, os.path.basename(f).replace(".csv", ".npz"))
        run_simulator(sim, scene, joint_list, f, out)

if __name__ == "__main__":
    main()
    simulation_app.close()