# Mimic Distill Walk (based on BeyondMimic+)

本项目是一套针对 **Unitree G1** 人形机器人的拟人化行走训练方案。核心逻辑采用两阶段法：首先通过 **Mimicry (模仿学习)** 捕获人类步态的高级特征，随后通过 **Distillation (策略蒸馏)** 提升策略的稳健性与部署效率。

> **核心目标**: 实现比传统 AMP (Adversarial Motion Priors) 更自然、更符合生物力学特征的步态，显著减少滑步与非自然抖动。

---

## 🛠️ 环境要求

- **Simulator**: [Isaac Lab](https://github.com/isaac-sim/IsaacLab) / Isaac Sim 4.0+
- **Base Framework**: [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking)
- **Robot**: Unitree G1 (29 DoF)
- **Environment**: Ubuntu 22.04 + ROS2 Humble

## 🚀 部署与安装

```bash
# 进入您的 Isaac Lab / beyondmimic 工作空间中的任务目录
cd <your_workspace_path>/source/legged_lab/legged_lab/tasks/

# 克隆本仓库（或确保 beyondmimicplus 文件夹在此目录下）
git clone [https://github.com/hanmingbai/mimic_distill_walk.git](https://github.com/hanmingbai/mimic_distill_walk.git)

本项目采用 Teacher-Student 蒸馏架构，请按顺序执行以下步骤：
1. 教师模型训练 (Teacher Training)

首先训练一个高质量的模仿学习策略作为教师模型。该阶段侧重于高精度的动作模仿。

# 在框架根目录下运行
python scripts/rsl_rl/train.py \
    --task Legged-Lab-BeyondMimicPlus-Flat-G1-v0 \
    --num_envs=4096 \
    --headless

2. 配置蒸馏参数 (Critical Step)

在启动学生模型训练前，必须手动指定教师模型的权重路径：

    配置文件: beyondmimicplusdistill/config/g1/agents/rsl_rl_distill_cfg.py

    操作: 找到 teacher.model_path，将其修改为您在第 1 步中生成的 .pt 文件绝对路径。

# 示例配置
teacher.model_path = "/home/{user_name}/legged_lab/logs/beyondmimicplus/model_xxx.pt"

3. 学生模型蒸馏 (Student Training)

执行蒸馏训练。学生模型将在教师模型的指导下，学习更稳健、更利于实机迁移的控制策略。

# 在框架根目录下运行
python scripts/rsl_rl/train.py \
    --task Legged-Lab-BeyondMimicPlusDistill-Flat-G1-v0 \
    --num_envs=4096 \
    --headless

部署与验证 (Deployment)
Sim-to-Sim 验证
在真机部署前，建议在以下物理引擎中进行交叉验证：

    Unitree Mujoco: 参考 unitree_mujoco 进行环境搭建。

    Deployment SDK: 适配 unitree-rl-lab 部署框架，实现从仿真到实机的平滑迁移。

模型导出

训练完成后，使用导出脚本将 .pt 权重转换为 ONNX 或部署所需的格式。

本研究基于 BeyondMimic 框架开发，感谢相关团队在全身运动追踪领域的开源贡献。

