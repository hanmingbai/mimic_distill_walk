# Mimic Distill Walk (based on BeyondMimic+)

> **核心目标**: achieve natural human-like walk instead of AMP (mode collaspe and too much reward engineering)

---

## 🛠️ 环境要求

- **Simulator**: [Isaac Lab 2.3.2](https://github.com/isaac-sim/IsaacLab) / Isaac Sim 5.1
- **Base Framework**: [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking)
- **Robot**: Unitree G1 (29 DoF)

## 🚀 部署与安装
```bash
git clone [https://github.com/hanmingbai/mimic_distill_walk.git](https://github.com/hanmingbai/mimic_distill_walk.git)
```
```bash
cd mimic_distill_walk
pip install -e .

cd rsl_rl
pip install -e .
```

本项目采用 Teacher-Student 蒸馏架构，请按顺序执行以下步骤：
1. 教师模型训练 (Teacher Training)

首先训练一个高质量的模仿学习策略作为教师模型。该阶段侧重于高精度的动作模仿。

# 在框架根目录下运行
```bash
python scripts/rsl_rl/train.py \
    --task Legged-Lab-BeyondMimicPlus-Flat-G1-v0 \
    --num_envs=4096 \
    --headless
```

2. 配置蒸馏参数 (Critical Step)

在启动学生模型训练前，必须手动指定教师模型的权重路径：

    配置文件: beyondmimicplusdistill/config/g1/agents/rsl_rl_distill_cfg.py

    操作: 找到 teacher.model_path，将其修改为您在第 1 步中生成的 .pt 文件绝对路径。

# 示例配置
```bash
teacher.model_path = "/home/{user_name}/legged_lab/logs/beyondmimicplus/model_xxx.pt"
```

3. 学生模型蒸馏 (Student Training)

执行蒸馏训练。学生模型将在教师模型的指导下，学习更稳健、更利于实机迁移的控制策略。

```bash
python scripts/rsl_rl/train.py \
    --task Legged-Lab-BeyondMimicPlusDistill-Flat-G1-v0 \
    --num_envs=4096 \
    --headless
```

# 部署与验证 (Deployment)
参考 unitree-rl-lab sim2sim部署框架
Sim-to-Sim 验证效果
```bash
https://www.bilibili.com/video/BV19WdaBHE5w/?share_source=copy_web&vd_source=7d8106d98f5362a5125feb535fa58925
```

# 感谢开源
本研究基于 BeyondMimic 框架开发，感谢相关团队在全身运动追踪领域的开源贡献。

