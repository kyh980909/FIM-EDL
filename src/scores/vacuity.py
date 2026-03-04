from __future__ import annotations

from torch import Tensor

from src.registry.scores import SCORE_REGISTRY


@SCORE_REGISTRY.register("vacuity")
def vacuity_score(alpha: Tensor, **kwargs) -> Tensor:
    k = alpha.size(1)
    s = alpha.sum(dim=1)
    return k / s
