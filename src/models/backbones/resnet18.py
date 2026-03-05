from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import resnet18

from src.registry.backbones import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("resnet18")
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        model = resnet18(weights=None if not pretrained else "DEFAULT")
        # Match CIFAR-style ResNet18 setup used in Fisher-EDL.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.out_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.net = model

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
