import gymnasium as gym

from . import agents, multi_tracking_distill_flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Legged-Lab-BeyondMimicPlusDistill-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multi_tracking_distill_flat_env_cfg.G1MultiTrackingDistillFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:G1FlatDAggerRunnerCfg",
    },
)

gym.register(
    id="Legged-Lab-BeyondMimicPlusDistill-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multi_tracking_distill_flat_env_cfg.G1MultiTrackingDistillFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:G1FlatDAggerRunnerCfg",
    },
)


gym.register(
    id="Legged-Lab-BeyondMimicPlusDistill-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multi_tracking_distill_flat_env_cfg.G1MultiTrackingDistillFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distill_cfg:G1FlatLowFreqDAggerRunnerCfg",
    },
)
