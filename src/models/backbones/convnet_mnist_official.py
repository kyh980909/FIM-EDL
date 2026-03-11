from __future__ import annotations

from torch import Tensor, nn

from src.registry.backbones import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("convnet_mnist_official")
class ConvNetMNISTOfficialBackbone(nn.Module):
    """MNIST conv backbone matching the official classical I-EDL setup."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        _ = pretrained
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.out_dim = 64

    def forward(self, x: Tensor) -> Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.classifier(z)
