from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class UIBugClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, backbone: str = "efficientnet_b0"):
        super().__init__()
        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights="DEFAULT")
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes),
            )
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT")
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


def get_model(num_classes: int = 8, backbone: str = "efficientnet_b0"):
    return UIBugClassifier(num_classes=num_classes, backbone=backbone)
