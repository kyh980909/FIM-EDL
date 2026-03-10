from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.registry.losses import LOSS_REGISTRY


def _edl_fit_per_sample(alpha: Tensor, target: Tensor) -> Tensor:
    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    y = F.one_hot(target, num_classes=alpha.size(1)).float()
    err = (y - p).pow(2)
    var = p * (1 - p) / (s + 1)
    return (err + var).sum(dim=1)


def _edl_fit(alpha: Tensor, target: Tensor) -> Tensor:
    return _edl_fit_per_sample(alpha, target).mean()


def _kl_dirichlet_to_uniform_per_sample(alpha: Tensor) -> Tensor:
    k = alpha.size(1)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    ln_b = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    ln_b_uni = torch.lgamma(torch.tensor(float(k), device=alpha.device)) - k * torch.lgamma(
        torch.tensor(1.0, device=alpha.device)
    )
    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)
    kl = ((alpha - 1) * (digamma_alpha - digamma_sum)).sum(dim=1, keepdim=True) + ln_b + ln_b_uni
    return kl.squeeze(1)


def _kl_dirichlet_to_uniform(alpha: Tensor) -> Tensor:
    return _kl_dirichlet_to_uniform_per_sample(alpha).mean()


@LOSS_REGISTRY.register("edl_fixed")
class EDLFixedLoss(nn.Module):
    def __init__(self, lam: float = 1.0, anneal_epochs: float = 10.0) -> None:
        super().__init__()
        self.lam = float(lam)
        self.anneal_epochs = float(anneal_epochs)

    def _lambda_weight(self, epoch: float) -> float:
        denom = max(self.anneal_epochs, 1e-6)
        progress = min(1.0, max(float(epoch), 0.0) / denom)
        return float(self.lam * progress)

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        epoch = float(kwargs.get("epoch", 0.0))
        lam = self._lambda_weight(epoch)
        fit = _edl_fit(alpha, target)
        y = F.one_hot(target, num_classes=alpha.size(1)).float()
        alpha_hat = alpha * (1.0 - y) + y
        reg = _kl_dirichlet_to_uniform(alpha_hat)
        total = fit + lam * reg
        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {
                "lambda_mean": float(lam),
                "lambda_min": float(lam),
                "lambda_max": float(lam),
                "lambda_std": 0.0,
                "lambda_target": float(self.lam),
                "lambda_progress": float(min(1.0, max(epoch, 0.0) / max(self.anneal_epochs, 1e-6))),
                "info": float("nan"),
                "info_std": float("nan"),
                "fisher_trace": float("nan"),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
