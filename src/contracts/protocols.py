from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class BackboneProtocol(Protocol):
    out_dim: int

    def __call__(self, x: Tensor) -> Tensor:
        ...


@runtime_checkable
class HeadProtocol(Protocol):
    def __call__(self, features: Tensor) -> Dict[str, Tensor]:
        ...


@runtime_checkable
class LossProtocol(Protocol):
    def __call__(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        ...


@runtime_checkable
class ScoreProtocol(Protocol):
    def __call__(self, alpha: Tensor, **kwargs: Any) -> Tensor:
        ...


def assert_module_instance(module: nn.Module, protocol: type[Protocol], name: str) -> None:
    if not isinstance(module, protocol):
        raise TypeError(f"{name} does not satisfy protocol {protocol.__name__}")
