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
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
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
        self.hidden = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.out_dim = 512
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0.0, (2.0 / n) ** 0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.hidden(z)
