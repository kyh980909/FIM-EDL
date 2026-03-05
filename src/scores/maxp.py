from __future__ import annotations

from torch import Tensor

from src.registry.scores import SCORE_REGISTRY


@SCORE_REGISTRY.register("maxp")
def maxp_score(alpha: Tensor, **kwargs) -> Tensor:
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    return probs.max(dim=1).values
