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


def create_llanet_model(cfg):
    """
    创建独立的LLANet模型实例

    Args:
        cfg: 配置参数
        detailed_loss_logger: 可选的分项损失记录器，用于记录 XYTL 分项损失
                             如果为None且cfg有output_dir属性，则自动创建logger

    Returns:
        模型配置
    """
    if cfg is None:
        raise ValueError("cfg 不能为空")
    # 1. 种子设置 (保持不变)
    seed = random.randint(1000, 9999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. 决定是否使用 ImageNet 预训练
    use_pretrained = cfg.get("use_pretrained_backbone", True)
    pretrained_model_name = cfg.get("pretrained_model_name", "mobilenetv4_conv_medium")
    feature_dim = cfg.get("featuremap_out_channel", 64)
    hidden_dim = cfg.get("fc_hidden_dim", feature_dim)

    model = L(LLANet)(
        backbone=L(TimmMobileNetV4Wrapper)(
            model_name=pretrained_model_name,
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
        ),
    )
    return model
