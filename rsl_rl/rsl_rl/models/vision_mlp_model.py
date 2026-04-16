# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from rsl_rl.models.mlp_model import MLPModel

class VisionMLPModel(MLPModel):
    """
    基于 MLPModel 修改的视觉模型。
    集成 3-layer CNN + GAP (Global Average Pooling) 提取器。
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        # [视觉蒸馏新增]
        num_proprio: int = 0,               # 本体感知维度
        visual_output_dim: int = 32,        # GAP 后的输出维度
    ) -> None:
        # 1. 记录视觉参数
        self.num_proprio = num_proprio
        self.visual_output_dim = visual_output_dim

        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        
        # 2. 定义 3-layer CNN + GAP 结构
        # 输入: [B, 1, 58, 87]
        self.cnn_encoder = nn.Sequential(
            # Layer 1: 58x87 -> 29x44 (Padding=1, Stride=2)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ELU(),
            # Layer 2: 29x44 -> 15x22
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            # Layer 3: 15x22 -> 8x11
            nn.Conv2d(32, visual_output_dim, kernel_size=3, stride=2, padding=1), nn.ELU(),
            # Global Average Pooling (GAP): 8x11 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # 变成 [B, 32]
        )

        print("<< --------------------VisionMLP Created --------------------- >>")

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """重写特征提取：手动切分拉平后的观测向量并经过 CNN。"""
        # 按照 obs_groups 顺序拼接原始观测
        # --- 兼容性逻辑开始 ---
        if isinstance(obs, TensorDict):
            # 正常训练/推理流程
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            full_obs = torch.cat(obs_list, dim=-1)
        else:
            # ONNX 导出流程：obs 已经是拼接好的 5142 维 Tensor
            full_obs = obs
        
        # [关键] 数据切分逻辑
        proprio_obs = full_obs[:, :self.num_proprio]
        depth_flat = full_obs[:, self.num_proprio:]
        
        # 1. 本体感知：使用父类的归一化器 (EmpiricalNormalization)
        proprio_latent = self.obs_normalizer(proprio_obs)
        
        # 2. 视觉感知：还原 4D 形状并过 CNN
        # 输入像素通常需要缩放，建议在环境配置中完成，或在此处 / 5.0
        depth_img = depth_flat.view(-1, 1, 58, 87)
        visual_latent = self.cnn_encoder(depth_img)
        
        # 3. 最终拼接送入 MLP Backbone
        return torch.cat([proprio_latent, visual_latent], dim=-1)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """覆盖维数计算：告诉父类归一化器只负责本体感知部分。"""
        active_obs_groups = obs_groups[obs_set]
        # 归一化器的维度仅为本体感知维度
        return active_obs_groups, self.num_proprio

    def _get_latent_dim(self) -> int:
        """覆盖 MLP 输入维度：本体感知 + 32维视觉特征。"""
        return self.num_proprio + self.visual_output_dim

    def update_normalization(self, obs: TensorDict) -> None:
        """仅更新本体感知部分的归一化统计量。"""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            full_obs = torch.cat(obs_list, dim=-1)
            proprio_obs = full_obs[:, :self.num_proprio]
            self.obs_normalizer.update(proprio_obs)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.mlp(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output