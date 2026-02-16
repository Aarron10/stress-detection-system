import torch
import torch.nn as nn
from torchvision import models

class StressModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        MobileNetV2 based model for stress detection.
        Args:
            num_classes (int): Number of output classes (e.g., 2 for Focused vs Stressed).
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """
        super(StressModel, self).__init__()
        
        # Load MobileNetV2
        self.backbone = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
        
        # MobileNetV2 classifier is a Sequential block:
        # (0): Dropout(p=0.2, inplace=False)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)
        
        # We replace the classifier to match our number of classes
        # The last channel of features is 1280 for MobileNetV2
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
