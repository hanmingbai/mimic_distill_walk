# Mimic Distill Walk

> **目标(The Goal)**: Achieve Natural Human-Like Walk Instead of AMP (Mode Collaspe and Too Much Reward Engineering)

---

## 环境(Environment)

- **Simulator**: [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html) / Isaac Sim 5.1
- **Base Framework**: [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking)
- **Robot**: Unitree G1 (29 DoF)

## 安装(Installation)
下载代码至本地
Download the package
```bash
git clone https://github.com/hanmingbai/mimic_distill_walk.git
```
安装legged_lab
Install legged_lab
```bash
cd mimic_distill_walk
pip install -e source/legged_lab
```
安装强化学习库
Install RSL_RL
```bash
cd rsl_rl
pip install -e .
```

## 训练(Training Pipeline)
本项目采用 Teacher-Student 蒸馏架构，请按顺序执行以下步骤：
This project uses a Teacher-Student distillation architecture. Please follow these steps in sequence:

### 教师模型训练 (Teacher Training)
首先训练一个高质量的模仿学习策略作为教师模型，该阶段侧重于高精度的多动作模仿，在`source/legged_lab/legged_lab/tasks/beyondmimicplus/config/g1/multi_tracking_flat_env_cfg.py`里修改数据集路径为`motion_dir = "/home/{user_name}/mimic_distill_walk/motions/seed_g1/A036/walk/003/"`，并在在根目录下运行：
First, train a high-quality imitation learning strategy as the teacher model. This stage focuses on high-precision multi-action imitation. In `source/legged_lab/legged_lab/tasks/beyondmimicplus/config/g1/multi_tracking_flat_env_cfg.py`, modify the dataset path to `motion_dir = "/home/{user_name}/mimic_distill_walk/motions/seed_g1/A036/walk/003/"`, and run the following in the root directory:
```bash
cd mimic_distill_walk
python scripts/rsl_rl/train.py --task Legged-Lab-BeyondMimicPlus-Flat-G1-v0 --num_envs=4096 --headless
```

### 手动配置蒸馏参数 (Critical Step)
在启动学生模型训练前，必须手动指定教师模型的权重路径，配置文件的路径是`source/legged_lab/legged_lab/tasks/beyondmimicplusdistill/config/g1/agents/rsl_rl_distill_cfg.py`，找到变量teacher.model_path，将其路径更改为在第一阶段训练好的模型路径
Before starting student model training, you must manually specify the weight path for the teacher model. The configuration file path is `source/legged_lab/legged_lab/tasks/beyondmimicplusdistill/config/g1/agents/rsl_rl_distill_cfg.py`. Locate the variable teacher.model_path and change its path to the path of the model trained in the first stage.
```bash
teacher.model_path = "/home/{user_name}/mimic_distill_walk/logs/rsl_rl/g1_flat_mimic_plus_with_seed_g1_A036_walk_003_datasets/2026-04-18_02-09-00/model_29999.pt"
```

### 学生模型蒸馏 (Student Training)
执行蒸馏训练，学生模型将在教师模型的指导下，将Tracker策略蒸馏为Velocity策略，在`source/legged_lab/legged_lab/tasks/beyondmimicplusdistill/config/g1/multi_tracking_distill_flat_env_cfg.py`里修改数据集路径为`motion_dir = "/home/{user_name}/mimic_distill_walk/motions/seed_g1/A036/walk/003/"`，然后执行：
To perform distillation training, the student model, guided by the teacher model, will distill the Tracker policy into a Velocity policy. In `source/legged_lab/legged_lab/tasks/beyondmimicplusdistill/config/g1/multi_tracking_distill_flat_env_cfg.py`, modify the dataset path to `motion_dir = "/home/{user_name}/mimic_distill_walk/motions/seed_g1/A036/walk/003/"`, and then execute:
```bash
cd mimic_distill_walk
python scripts/rsl_rl/train.py --task Legged-Lab-BeyondMimicPlusDistill-Flat-G1-v0 --num_envs=4096 --headless
```

## 部署 (Deployment)
### Sim-to-Sim
参考 unitree-rl-lab里的sim-to-sim部署框架，仿真结果如视频所示
```bash
https://www.bilibili.com/video/BV19WdaBHE5w/?share_source=copy_web&vd_source=7d8106d98f5362a5125feb535fa58925
```

## 致谢
本项目基于 BeyondMimic/Isaac Lab/RSL_RL 开发。

