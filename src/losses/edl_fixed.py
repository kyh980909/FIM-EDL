from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.registry.losses import LOSS_REGISTRY


def _edl_fit(alpha: Tensor, target: Tensor) -> Tensor:
    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    y = F.one_hot(target, num_classes=alpha.size(1)).float()
    err = (y - p).pow(2)
    var = p * (1 - p) / (s + 1)
    return (err + var).sum(dim=1).mean()


def _kl_dirichlet_to_uniform(alpha: Tensor) -> Tensor:
    k = alpha.size(1)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    ln_b = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    ln_b_uni = torch.lgamma(torch.tensor(float(k), device=alpha.device)) - k * torch.lgamma(
        torch.tensor(1.0, device=alpha.device)
    )
    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    kl = ((alpha - 1) * (digamma_alpha - digamma_sum)).sum(dim=1, keepdim=True) + ln_b + ln_b_uni
    return kl.mean()


@LOSS_REGISTRY.register("edl_fixed")
class EDLFixedLoss(nn.Module):
    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        fit = _edl_fit(alpha, target)
        reg = _kl_dirichlet_to_uniform(alpha)
        total = fit + self.lam * reg
        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {"lambda_mean": float(self.lam), "lambda_min": float(self.lam), "lambda_max": float(self.lam)},
            "schema_version": LOSS_SCHEMA_VERSION,
        }
