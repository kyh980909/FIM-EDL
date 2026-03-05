from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.losses.edl_fixed import _edl_fit_per_sample, _kl_dirichlet_to_uniform_per_sample
from src.registry.losses import LOSS_REGISTRY


def _fisher_trace(alpha: Tensor) -> Tensor:
    # Trace(I(alpha)) approximation based on trigamma terms.
    trigamma_alpha = torch.polygamma(1, alpha)
    trigamma_sum = torch.polygamma(1, alpha.sum(dim=1, keepdim=True))
    return (trigamma_alpha - trigamma_sum).sum(dim=1)


@LOSS_REGISTRY.register("info_edl")
class EDLInfoAdaptiveLoss(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        fit_ps = _edl_fit_per_sample(alpha, target)
        kl_ps = _kl_dirichlet_to_uniform_per_sample(alpha)
        fit = fit_ps.mean()
        reg = kl_ps.mean()
        v = _fisher_trace(alpha).detach()
        lam = self.beta * torch.exp(-self.gamma * v)
        total = fit + (lam * kl_ps).mean()
        return {
            "total": total,
            "fit": fit,
            "reg": reg,
            "aux": {
                "lambda_mean": float(lam.mean().item()),
                "lambda_min": float(lam.min().item()),
                "lambda_max": float(lam.max().item()),
                "lambda_std": float(lam.std(unbiased=False).item()),
                "info": float(v.mean().item()),
                "info_std": float(v.std(unbiased=False).item()),
                "fim_trace_mean": float(v.mean().item()),
                "fisher_trace": float(v.mean().item()),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
