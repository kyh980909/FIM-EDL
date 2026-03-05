from __future__ import annotations

from torch import Tensor, nn

from src.registry.backbones import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("convnet")
class ConvNetBackbone(nn.Module):
    """Paper-style small ConvNet backbone for MNIST experiments."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        _ = pretrained  # Unused, kept for a unified backbone constructor signature.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = 256

    def forward(self, x: Tensor) -> Tensor:
        z = self.features(x)
        return z.flatten(1)
