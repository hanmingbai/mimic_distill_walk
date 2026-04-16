from isaaclab.utils import configclass

from legged_lab.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from legged_lab.tasks.beyondmimicplusdistill.config.g1.agents.rsl_rl_distill_cfg import LOW_FREQ_SCALE
from legged_lab.tasks.beyondmimicplusdistill.multi_tracking_distill_env_cfg import MultiTrackingDistillEnvCfg

import glob
import os


@configclass
class G1MultiTrackingDistillFlatEnvCfg(MultiTrackingDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # self.commands.motion.motion_file = [
        #                                         "/home/hmbai/legged_lab/motions/walk1_subject1_0s_45s.npz",
        #                                         # "/home/hmbai/legged_lab/motions/walk1_subject1_81.2s_86.7s.npz",
        #                                         # "/home/hmbai/legged_lab/motions/walk1_subject5_146.7s_159s.npz"
        #                                         # "/home/hmbai/legged_lab/motions/walk1_subject5_206.7s_263.7s.npz"
        #                                     ]
        
        motion_dir = "/home/hmbai/legged_lab/motions/seed_g1/A036/walk/003/"
        self.commands.motion.motion_file = glob.glob(os.path.join(motion_dir, "*.npz"))


@configclass
class G1MultiTrackingDistillFlatWoStateEstimationEnvCfg(G1MultiTrackingDistillFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1MultiTrackingDistillFlatLowFreqEnvCfg(G1MultiTrackingDistillFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


# Deploy
# conda activate isaaclab
# cd legged_lab/deploy/robots/g1_29dof/build
# ./g1_ctrl