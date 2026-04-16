from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
@configclass
class DAggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # 显式增加你在 ppo.py 中新增的参数名
    dagger_loss_mix_ratio: float = 0.5

@configclass
class G1FlatDAggerRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 250
    experiment_name = "g1_flat_dagger_ppo" # 建议修改实验名称以区分
    empirical_normalization = True

    # # 1. 网络结构配置 (Student)
    # actor = RslRlMLPModelCfg(
    #     class_name="MLPModel",
    #     hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     obs_normalization=True, # 取代原来的 empirical_normalization
    #     init_noise_std=1.0,
    # )
    
    # critic = RslRlMLPModelCfg(
    #     class_name="MLPModel",
    #     hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     obs_normalization=True,
    # )

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # 2. DAgger-PPO 算法配置
    algorithm = DAggerAlgorithmCfg(
        class_name="DAggerPPO", 
        
        # PPO 基础参数保持不变
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    teacher.model_path = "/home/hmbai/legged_lab/logs/rsl_rl/g1_flat_mimic_plus_with_seed_g1_A036_walk_003_datasets/2026-04-14_22-56-17/model_29999.pt"

    obs_groups = {
        "actor": ["policy"],      # 学生看本体感知信息
        "critic": ["critic"],     # Critic 看包含参考动作的信息
        "teacher": ["teacher"]     # Teacher 也看包含参考动作的信息
    }


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqDAggerRunnerCfg(G1FlatDAggerRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
