from __future__ import annotations

from omegaconf import DictConfig

from src.registry.backbones import BACKBONE_REGISTRY
from src.registry.heads import HEAD_REGISTRY
from src.registry.losses import LOSS_REGISTRY
from src.registry.scores import SCORE_REGISTRY


LOSS_ALLOWED_KEYS = {
    "edl_fixed": {"name", "lambda_value", "anneal_epochs"},
    "info_edl": {"name", "beta", "gamma", "info_type", "gate_type", "detach_weight", "objective"},
    "iedl_ref": {"name", "lambda_kl", "lambda_logdet", "fisher_c", "kl_anneal_epochs"},
}

LOSS_REQUIRED_KEYS = {
    "edl_fixed": {"name", "lambda_value", "anneal_epochs"},
    "info_edl": {"name", "beta", "gamma"},
    "iedl_ref": {"name", "lambda_kl", "lambda_logdet", "fisher_c", "kl_anneal_epochs"},
}


def validate_registry_bindings(cfg: DictConfig) -> None:
    BACKBONE_REGISTRY.get(cfg.model.backbone)
    HEAD_REGISTRY.get(cfg.model.head)
    LOSS_REGISTRY.get(cfg.loss.name)
    SCORE_REGISTRY.get(cfg.score.name)

    loss_name = str(cfg.loss.name)
    allowed = LOSS_ALLOWED_KEYS.get(loss_name)
    required = LOSS_REQUIRED_KEYS.get(loss_name)
    if allowed is None or required is None:
        raise ValueError(f"Missing validator spec for loss '{loss_name}'")

    present = set(cfg.loss.keys())
    extra = sorted(present - allowed)
    missing = sorted(required - present)
    if extra:
        raise ValueError(f"Unused loss config keys for '{loss_name}': {', '.join(extra)}")
    if missing:
        raise ValueError(f"Missing required loss config keys for '{loss_name}': {', '.join(missing)}")
