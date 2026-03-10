from __future__ import annotations

from torch import Tensor, nn

from src.registry.backbones import BACKBONE_REGISTRY


def _make_layers(cfg: list[int | str], batch_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(value)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        in_channels = out_channels
    return nn.Sequential(*layers)


@BACKBONE_REGISTRY.register("vgg16")
class VGG16Backbone(nn.Module):
    """CIFAR-style VGG16-BN feature extractor."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("CIFAR VGG16 backbone does not support pretrained=True")
        self.features = _make_layers(
            [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        z = self.features(x)
        z = self.pool(z)
        return z.flatten(1)
