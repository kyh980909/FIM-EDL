from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

# Register components by import side effects.
import src.models.backbones.resnet18  # noqa: F401
import src.models.heads.edl_head  # noqa: F401
import src.losses.edl_fixed  # noqa: F401
import src.losses.edl_info_adaptive  # noqa: F401
import src.scores.vacuity  # noqa: F401

from src.contracts.protocols import (
    BackboneProtocol,
    HeadProtocol,
    LossProtocol,
    ScoreProtocol,
    assert_module_instance,
)
from src.registry.backbones import BACKBONE_REGISTRY
from src.registry.heads import HEAD_REGISTRY
from src.registry.losses import LOSS_REGISTRY
from src.registry.scores import SCORE_REGISTRY


class InfoEDLLightningModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        backbone_cls = BACKBONE_REGISTRY.get(cfg.model.backbone)
        self.backbone: nn.Module = backbone_cls(pretrained=cfg.model.pretrained)
        assert_module_instance(self.backbone, BackboneProtocol, "backbone")

        head_cls = HEAD_REGISTRY.get(cfg.model.head)
        self.head: nn.Module = head_cls(
            in_dim=self.backbone.out_dim,
            num_classes=cfg.model.num_classes,
            evidence_fn=cfg.model.evidence_fn,
        )
        assert_module_instance(self.head, HeadProtocol, "head")

        loss_cls = LOSS_REGISTRY.get(cfg.loss.name)
        if cfg.loss.name == "edl_fixed":
            self.loss_fn: nn.Module = loss_cls(lam=cfg.loss.lambda_value)
        else:
            self.loss_fn = loss_cls(beta=cfg.loss.beta, gamma=cfg.loss.gamma)
        assert_module_instance(self.loss_fn, LossProtocol, "loss")

        self.score_fn = SCORE_REGISTRY.get(cfg.score.name)
        if not callable(self.score_fn):
            raise TypeError("score function must be callable")

        self._shape_checked = False

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feat = self.backbone(x)
        out = self.head(feat)
        uncertainty = self.score_fn(out["alpha"])
        out["uncertainty_score"] = uncertainty
        return out

    def _assert_shapes(self, batch: Any, out: Dict[str, Tensor]) -> None:
        x, y = batch
        bsz = x.size(0)
        assert out["alpha"].shape == (bsz, self.cfg.model.num_classes)
        assert out["probs"].shape == (bsz, self.cfg.model.num_classes)
        assert y.ndim == 1

    def _common_step(self, batch: Any, stage: str) -> Tensor:
        x, y = batch
        out = self(x)
        if not self._shape_checked:
            self._assert_shapes(batch, out)
            self._shape_checked = True
        loss_out = self.loss_fn(out["alpha"], y)
        pred = out["probs"].argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log(f"{stage}/loss", loss_out["total"], prog_bar=(stage != "train"), on_epoch=True)
        self.log(f"{stage}/fit", loss_out["fit"], on_epoch=True)
        self.log(f"{stage}/reg", loss_out["reg"], on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True)
        for key, value in loss_out["aux"].items():
            self.log(f"{stage}/aux/{key}", value, on_epoch=True)
        return loss_out["total"]

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._common_step(batch, "test")

    def predict_scores(self, batch: Any) -> Tensor:
        x, _ = batch
        out = self(x)
        return out["uncertainty_score"]

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            momentum=self.cfg.optimizer.momentum,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.cfg.trainer.max_epochs)
        return {"optimizer": optim, "lr_scheduler": sched}
