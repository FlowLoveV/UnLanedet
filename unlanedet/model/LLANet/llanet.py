"""
LLANet Model Definition

Combines MobileNetV4-Small backbone, GSA-FPN neck, and LLANetHead into a single model.
"""

import torch.nn as nn

from .gsa_fpn import GSAFPN
from .llanet_head import LLANetHead


class LLANet(nn.Module):
    """LLANet model combining MobileNetV4-Small backbone, GSA-FPN neck, and LLANetHead."""

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, data):
        img = data["img"]  # Extract image tensor from data dict

        # Ensure image is on the same device as the model
        device = next(self.backbone.parameters()).device
        if img.device != device:
            img = img.to(device)

        # DEBUG: Check input image stats
        import torch

        if torch.isnan(img).any():
            print("NaN detected in input image!", flush=True)
        print(
            f"DEBUG Input Image Stats: Min={img.min().item()}, Max={img.max().item()}, Mean={img.mean().item()}",
            flush=True,
        )

        features = self.backbone(img)
        neck_features = self.neck(features)

        if self.training:
            # During training, forward returns outputs and losses
            outputs = self.head(neck_features)
            current_iter = data.get("iter", 0)
            losses = self.head.loss(outputs, data, current_iter=current_iter)
            return losses
        else:
            # During inference, forward returns predictions
            return self.head(neck_features)

    def get_lanes(self, output):
        return self.head.get_lanes(output)
