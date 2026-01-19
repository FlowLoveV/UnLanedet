"""
Timm MobileNetV4 Backbone Wrapper
"""

import torch
import torch.nn as nn
import timm


class TimmMobileNetV4Wrapper(nn.Module):
    """timm MobileNetV4 wrapper to match our interface."""

    def __init__(self, model_name='mobilenetv4_conv_small.e2400_r224_in1k',
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
        print(f"[TimmMobileNetV4] Using {model_name}, pretrained={pretrained}")
        print(f"[TimmMobileNetV4] Output channels: {self.out_channels}")

    def forward(self, x):
        return self.model(x)
