from __future__ import annotations

from omegaconf import DictConfig

from src.registry.backbones import BACKBONE_REGISTRY
from src.registry.heads import HEAD_REGISTRY
from src.registry.losses import LOSS_REGISTRY
from src.registry.scores import SCORE_REGISTRY


def validate_registry_bindings(cfg: DictConfig) -> None:
    BACKBONE_REGISTRY.get(cfg.model.backbone)
    HEAD_REGISTRY.get(cfg.model.head)
    LOSS_REGISTRY.get(cfg.loss.name)
    SCORE_REGISTRY.get(cfg.score.name)
