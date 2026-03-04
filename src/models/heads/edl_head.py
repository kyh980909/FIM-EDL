from __future__ import annotations

from typing import Dict

from torch import Tensor, nn
import torch.nn.functional as F

from src.registry.heads import HEAD_REGISTRY


@HEAD_REGISTRY.register("edl")
class EDLHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, evidence_fn: str = "softplus") -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.evidence_fn = evidence_fn

    def _evidence(self, logits: Tensor) -> Tensor:
        if self.evidence_fn == "relu":
            return F.relu(logits)
        return F.softplus(logits)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.fc(features)
        evidence = self._evidence(logits)
        alpha = evidence + 1.0
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        return {"logits": logits, "evidence": evidence, "alpha": alpha, "probs": probs}
