from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.losses.edl_fixed import _kl_dirichlet_to_uniform_per_sample
from src.registry.losses import LOSS_REGISTRY


def _fisher_diag(alpha: Tensor) -> Tensor:
    s = alpha.sum(dim=1, keepdim=True)
    return torch.polygamma(1, alpha) - torch.polygamma(1, s)


def _i_mse_per_sample(alpha: Tensor, target: Tensor) -> Tensor:
    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    y = F.one_hot(target, num_classes=alpha.size(1)).float()
    err = (y - p).pow(2)
    var = p * (1 - p) / (s + 1.0)
    w = _fisher_diag(alpha).clamp_min(1e-8)
    return (w * (err + var)).sum(dim=1)


def _fisher_logdet_per_sample(alpha: Tensor) -> Tensor:
    bsz, k = alpha.shape
    diag_term = torch.diag_embed(torch.polygamma(1, alpha))
    sum_term = torch.polygamma(1, alpha.sum(dim=1, keepdim=True))
    ones = torch.ones((bsz, k, k), device=alpha.device, dtype=alpha.dtype)
    fim = diag_term - sum_term.unsqueeze(-1) * ones
    jitter = 1e-6 * torch.eye(k, device=alpha.device, dtype=alpha.dtype).unsqueeze(0)
    sign, logabsdet = torch.linalg.slogdet(fim + jitter)
    safe = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -20.0))
    return safe


@LOSS_REGISTRY.register("iedl_ref")
class IEDLRefLoss(nn.Module):
    def __init__(self, lambda_kl: float = 1.0, lambda_logdet: float = 0.01) -> None:
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_logdet = lambda_logdet

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        i_mse_ps = _i_mse_per_sample(alpha, target)
        i_mse = i_mse_ps.mean()

        y = F.one_hot(target, num_classes=alpha.size(1)).float()
        alpha_hat = alpha * (1.0 - y) + y
        kl_ps = _kl_dirichlet_to_uniform_per_sample(alpha_hat)
        kl = kl_ps.mean()

        logdet_ps = _fisher_logdet_per_sample(alpha)
        logdet_mean = logdet_ps.mean()

        # Ref-style objective: I-MSE + lambda_kl * KL - lambda_logdet * log|I|.
        total = i_mse + self.lambda_kl * kl - self.lambda_logdet * logdet_mean

        return {
            "total": total,
            "fit": i_mse,
            "reg": kl,
            "aux": {
                "lambda_mean": float(self.lambda_kl),
                "lambda_min": float(self.lambda_kl),
                "lambda_max": float(self.lambda_kl),
                "lambda_std": 0.0,
                "info": float(_fisher_diag(alpha).mean().detach().item()),
                "info_std": float(_fisher_diag(alpha).std(unbiased=False).detach().item()),
                "fisher_trace": float(_fisher_diag(alpha).sum(dim=1).mean().detach().item()),
                "logdet_fim": float(logdet_mean.detach().item()),
                "i_mse": float(i_mse.detach().item()),
                "kl_weight": float(self.lambda_kl),
                "logdet_weight": float(self.lambda_logdet),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
