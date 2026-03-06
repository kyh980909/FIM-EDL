from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.contracts.schemas import LOSS_SCHEMA_VERSION
from src.losses.edl_fixed import _kl_dirichlet_to_uniform_per_sample
from src.registry.losses import LOSS_REGISTRY


def _fisher_terms_per_sample(alpha: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """IEDL reference terms from code_classical/models/FisherEvidentialN.py."""
    s = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(target, num_classes=alpha.size(1)).float()
    gamma1_alpha = torch.polygamma(1, alpha)
    gamma1_s = torch.polygamma(1, s)
    gap = y - alpha / s

    loss_mse = (gap.pow(2) * gamma1_alpha).sum(dim=1)
    loss_var = (alpha * (s - alpha) * gamma1_alpha / (s * s * (s + 1.0))).sum(dim=1)

    eps = 1e-8
    ratio_sum = (gamma1_s / gamma1_alpha.clamp_min(eps)).sum(dim=1)
    inner = (1.0 - ratio_sum).clamp_min(eps)
    loss_det_fisher = -(torch.log(gamma1_alpha.clamp_min(eps)).sum(dim=1) + torch.log(inner))
    return loss_mse, loss_var, loss_det_fisher


def _fisher_diag(alpha: Tensor) -> Tensor:
    s = alpha.sum(dim=1, keepdim=True)
    return torch.polygamma(1, alpha) - torch.polygamma(1, s)


@LOSS_REGISTRY.register("iedl_ref")
class IEDLRefLoss(nn.Module):
    def __init__(
        self,
        lambda_kl: float = -1.0,
        fisher_c: float = 0.05,
        kl_anneal_epochs: float = 10.0,
        lambda_logdet: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_kl = lambda_kl
        # Keep backward compatibility with old config key while preferring fisher_c.
        self.fisher_c = float(fisher_c if lambda_logdet is None else lambda_logdet)
        self.kl_anneal_epochs = float(kl_anneal_epochs)

    def _kl_weight(self, epoch: float) -> float:
        # IEDL reference uses regr=min(1.0, epoch/10.0) when kl_c == -1.
        if self.lambda_kl >= 0.0:
            return float(self.lambda_kl)
        denom = max(self.kl_anneal_epochs, 1e-6)
        return float(min(1.0, max(epoch, 0.0) / denom))

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        epoch = float(kwargs.get("epoch", 0.0))
        kl_weight = self._kl_weight(epoch)

        loss_mse_ps, loss_var_ps, loss_det_ps = _fisher_terms_per_sample(alpha, target)
        loss_mse = torch.where(torch.isfinite(loss_mse_ps), loss_mse_ps, torch.zeros_like(loss_mse_ps)).mean()
        loss_var = torch.where(torch.isfinite(loss_var_ps), loss_var_ps, torch.zeros_like(loss_var_ps)).mean()
        loss_det = torch.where(torch.isfinite(loss_det_ps), loss_det_ps, torch.zeros_like(loss_det_ps)).mean()

        y = F.one_hot(target, num_classes=alpha.size(1)).float()
        alpha_hat = alpha * (1.0 - y) + y
        kl_ps = _kl_dirichlet_to_uniform_per_sample(alpha_hat)
        kl = torch.where(torch.isfinite(kl_ps), kl_ps, torch.zeros_like(kl_ps)).mean()
        total = loss_mse + loss_var + self.fisher_c * loss_det + kl_weight * kl

        return {
            "total": total,
            "fit": loss_mse + loss_var,
            "reg": kl,
            "aux": {
                "lambda_mean": float(kl_weight),
                "lambda_min": float(kl_weight),
                "lambda_max": float(kl_weight),
                "lambda_std": 0.0,
                "info": float(_fisher_diag(alpha).mean().detach().item()),
                "info_std": float(_fisher_diag(alpha).std(unbiased=False).detach().item()),
                "fisher_trace": float(_fisher_diag(alpha).sum(dim=1).mean().detach().item()),
                "i_mse": float(loss_mse.detach().item()),
                "var": float(loss_var.detach().item()),
                "loss_fisher": float(loss_det.detach().item()),
                "kl_weight": float(kl_weight),
                "fisher_c": float(self.fisher_c),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
