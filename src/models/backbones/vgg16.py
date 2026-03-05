from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import vgg16

from src.registry.backbones import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("vgg16")
class VGG16Backbone(nn.Module):
    """VGG16 feature extractor used in I-EDL CIFAR10 setting."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        model = vgg16(weights="DEFAULT" if pretrained else None)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512

    def forward(self, x: Tensor) -> Tensor:
        z = self.features(x)
        z = self.pool(z)
        return z.flatten(1)
