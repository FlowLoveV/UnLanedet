"""
简单模型工厂 - 专门用于生成独立的LLANet模型实例
避免多个训练实例间的冲突
"""

import os
import random
import numpy as np
import torch
import timm
from unlanedet.config import LazyCall as L
from unlanedet.model.LLANet.llanet_head import LLANetHead
from unlanedet.model.LLANet.gsa_fpn import GSAFPN
from unlanedet.model.LLANet.llanet import LLANet


class TimmMobileNetV4Wrapper(torch.nn.Module):
    """timm MobileNetV4 wrapper to match our interface."""

    def __init__(
        self,
        model_name="mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=True,
        features_only=True,
        out_indices=[2, 3, 4],
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            out_indices=out_indices,
        )

    def forward(self, x):
        return self.model(x)


def create_llanet_model(cfg):
    """
    创建独立的LLANet模型实例

    Args:
        cfg: 配置参数

    Returns:
        模型配置
    """
    # 生成唯一随机种子确保独立性
    seed = random.randint(1000, 9999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 决定是否使用 ImageNet 预训练
    use_pretrained = getattr(cfg, "use_pretrained_backbone", True)  # 默认使用预训练

    # 创建模型 - 使用 timm 的 MobileNetV4
    model = L(LLANet)(
        backbone=L(TimmMobileNetV4Wrapper)(
            model_name="mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=use_pretrained,
            features_only=True,
            out_indices=[2, 3, 4],  # 输出 stride 8, 16, 32 的特征
        ),
        neck=L(GSAFPN)(
            in_channels=[64, 96, 960],  # timm mobilenetv4_conv_small 的输出通道
            out_channels=64,
            num_outs=3,
            scm_kernel_size=3,
            enable_global_semantic=True,
        ),
        head=L(LLANetHead)(
            num_priors=cfg.num_priors,
            refine_layers=3,
            fc_hidden_dim=64,
            sample_points=36,
            cfg=cfg,
            enable_category=True,
            enable_attribute=True,
        ),
    )

    return model
