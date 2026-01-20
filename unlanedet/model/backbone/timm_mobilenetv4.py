"""
Timm MobileNetV4 Backbone Wrapper
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


class TimmMobileNetV4Wrapper(nn.Module):
    """timm MobileNetV4 wrapper to match our interface."""

    def __init__(self, model_name='mobilenetv4_conv_medium',
                 pretrained=True, features_only=True, out_indices=[2, 3, 4]):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            out_indices=out_indices,
        )

        # 获取输出通道数
        self.out_channels = self.model.feature_info.channels()
        logger.info(f"[TimmMobileNetV4] Using {model_name}, pretrained={pretrained}")
        logger.info(f"[TimmMobileNetV4] Output channels: {self.out_channels}")
        logger.info(f"[TimmMobileNetV4] Actual model class: {type(self.model).__name__}")

    def forward(self, x):
        return self.model(x)
