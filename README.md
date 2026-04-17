# Mimic Distill Walk

> **核心目标**: achieve natural human-like walk instead of AMP (mode collaspe and too much reward engineering)

---

## 环境

- **Simulator**: [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html) / Isaac Sim 5.1
- **Base Framework**: [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking)
- **Robot**: Unitree G1 (29 DoF)

## 安装
下载代码至本地
```bash
git clone https://github.com/hanmingbai/mimic_distill_walk.git
```
安装legged_lab
```bash
cd mimic_distill_walk
pip install -e source/legged_lab
```
安装强化学习库
```bash
cd rsl_rl
pip install -e .
```

## 训练
本项目采用 Teacher-Student 蒸馏架构，请按顺序执行以下步骤：

### 教师模型训练 (Teacher Training)
首先训练一个高质量的模仿学习策略作为教师模型，该阶段侧重于高精度的多动作模仿，在根目录下运行：
```bash
cd mimic_distill_walk
python scripts/rsl_rl/train.py --task Legged-Lab-BeyondMimicPlus-Flat-G1-v0 --num_envs=4096 --headless
```

### 手动配置蒸馏参数 (Critical Step)
在启动学生模型训练前，必须手动指定教师模型的权重路径，配置文件的路径是source/legged_lab/legged_lab/tasks/beyondmimicplusdistill/config/g1/agents/rsl_rl_distill_cfg.py，找到变量teacher.model_path，将其路径更改为在第一阶段训练好的模型路径
```bash
teacher.model_path = "/home/{user_name}/mimic_distill_walk/logs/rsl_rl/g1_flat_mimic_plus_with_seed_g1_A036_walk_003_datasets/2026-04-18_02-09-00/model_29999.pt"
```

### 学生模型蒸馏 (Student Training)
执行蒸馏训练，学生模型将在教师模型的指导下，将Tracker策略蒸馏为Velocity策略
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

