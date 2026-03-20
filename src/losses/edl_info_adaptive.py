from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
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
    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 1.0,
        info_type: str = "fisher",
        gate_type: str = "exp",
        detach_weight: bool = True,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.info_type = str(info_type)
        self.gate_type = str(gate_type)
        self.detach_weight = bool(detach_weight)

    def _control_signal(self, alpha: Tensor, target: Tensor) -> Tensor:
        info_type = self.info_type.lower()
        if info_type == "fisher":
            return _fisher_trace(alpha)
        if info_type == "alpha0":
            return alpha.sum(dim=1)
        if info_type == "target_alpha":
            y = F.one_hot(target, num_classes=alpha.size(1)).float()
            return (alpha * y).sum(dim=1)
        raise ValueError(f"Unsupported info_type: {self.info_type}")

    def _lambda_weight(self, signal: Tensor) -> Tensor:
        gate_type = self.gate_type.lower()
        if gate_type == "exp":
            return self.beta * torch.exp(-self.gamma * signal)
        if gate_type == "constant":
            return torch.full_like(signal, fill_value=self.beta)
        raise ValueError(f"Unsupported gate_type: {self.gate_type}")

    def forward(self, alpha: Tensor, target: Tensor, **kwargs: Any) -> Dict[str, Any]:
        fit_ps = _edl_fit_per_sample(alpha, target)
        kl_ps = _kl_dirichlet_to_uniform_per_sample(alpha)
        fit = fit_ps.mean()
        reg = kl_ps.mean()
        signal = self._control_signal(alpha=alpha, target=target)
        if self.detach_weight:
            signal = signal.detach()
        lam = self._lambda_weight(signal)
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
                "info": float(signal.mean().detach().item()),
                "info_std": float(signal.std(unbiased=False).detach().item()),
                "fim_trace_mean": float(_fisher_trace(alpha).mean().detach().item()),
                "fisher_trace": float(_fisher_trace(alpha).mean().detach().item()),
                "detach_weight": float(self.detach_weight),
            },
            "schema_version": LOSS_SCHEMA_VERSION,
        }
