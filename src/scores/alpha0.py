from __future__ import annotations

from torch import Tensor

from src.registry.scores import SCORE_REGISTRY


@SCORE_REGISTRY.register("alpha0")
def alpha0_score(alpha: Tensor, **kwargs) -> Tensor:
    return alpha.sum(dim=1)
