"""
简单模型工厂 - 专门用于生成独立的LLANet模型实例
避免多个训练实例间的冲突
"""

import os
import random
import numpy as np
import torch
from unlanedet.config import LazyCall as L
from unlanedet.model.backbone.timm_mobilenetv4 import TimmMobileNetV4Wrapper
from unlanedet.model.LLANet.llanet_head import LLANetHead
from unlanedet.model.LLANet.gsa_fpn import GSAFPN
from unlanedet.model.LLANet.llanet import LLANet


def create_llanet_model(cfg, detailed_loss_logger=None):
    """
    创建独立的LLANet模型实例

    Args:
        cfg: 配置参数
        detailed_loss_logger: 可选的分项损失记录器，用于记录 XYTL 分项损失
                             如果为None且cfg有output_dir属性，则自动创建logger

    Returns:
        模型配置
    """
    # 1. 种子设置 (保持不变)
    seed = random.randint(1000, 9999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. 决定是否使用 ImageNet 预训练
    use_pretrained = getattr(cfg, "use_pretrained_backbone", True)

    # 3. Logger 设置 (保持不变)
    if detailed_loss_logger is None and hasattr(cfg, "output_dir"):
        from unlanedet.utils.detailed_loss_logger import DetailedLossLogger

        detailed_loss_logger = DetailedLossLogger(
            output_dir=cfg.output_dir, filename="detailed_metrics.json"
        )

    # 读取特征图输出通道数 (Neck的输出，Head的输入)
    feature_dim = getattr(cfg, "featuremap_out_channel", 64)
    cfg.featuremap_out_channel = feature_dim

    # 读取全连接层隐藏层维度 (Head 内部)
    hidden_dim = getattr(cfg, "fc_hidden_dim", feature_dim)

    # 创建模型
    model = L(LLANet)(
        backbone=L(TimmMobileNetV4Wrapper)(
            model_name="mobilenetv4_conv_medium",
            pretrained=use_pretrained,
            features_only=True,
            out_indices=[2, 3, 4],
        ),
        neck=L(GSAFPN)(
            in_channels=[80, 160, 960],
            out_channels=feature_dim,
            num_outs=3,
            scm_kernel_size=3,
            enable_global_semantic=True,
        ),
        head=L(LLANetHead)(
            num_priors=cfg.num_priors,
            refine_layers=3,
            prior_feat_channels=feature_dim,
            fc_hidden_dim=hidden_dim,
            sample_points=36,
            cfg=cfg,
            enable_category=True,
            enable_attribute=True,
            detailed_loss_logger=detailed_loss_logger,
        ),
    )

    return model
