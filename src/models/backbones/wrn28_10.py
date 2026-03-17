from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from src.registry.backbones import BACKBONE_REGISTRY


class _WideBasic(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out


class _WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10, drop_rate: float = 0.0) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WRN depth must satisfy (depth-4)%6==0")
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_layer(widths[0], widths[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_layer(widths[1], widths[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_layer(widths[2], widths[3], n, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = widths[3]

    def _make_layer(self, in_planes: int, planes: int, n: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = [_WideBasic(in_planes, planes, stride=stride, drop_rate=drop_rate)]
        for _ in range(1, n):
            layers.append(_WideBasic(planes, planes, stride=1, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        return out.flatten(1)


@BACKBONE_REGISTRY.register("wrn28_10")
class WRN28x10Backbone(nn.Module):
    """WideResNet-28-10 style backbone for few-shot setting."""

    def __init__(self, pretrained: bool = False, checkpoint_path: str = "") -> None:
        super().__init__()
        self.net = _WideResNet(depth=28, widen_factor=10, drop_rate=0.0)
        self.out_dim = self.net.out_dim
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        elif pretrained:
            raise ValueError(
                "wrn28_10 pretrained=True requires model.backbone_checkpoint because the repo does not bundle the Yang et al. (2021) weights."
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        cleaned = {}
        for key, value in state_dict.items():
            key = str(key)
            if key.startswith("backbone.net."):
                cleaned[key[len("backbone.net.") :]] = value
            elif key.startswith("net."):
                cleaned[key[len("net.") :]] = value
            elif key.startswith("backbone."):
                cleaned[key[len("backbone.") :]] = value
            else:
                cleaned[key] = value
        missing, unexpected = self.net.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected WRN checkpoint keys: {unexpected[:10]}")
        if missing:
            raise ValueError(
                "WRN checkpoint is incomplete for the backbone. Missing keys start with: "
                f"{missing[:10]}"
            )
