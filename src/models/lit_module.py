from __future__ import annotations

import inspect
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

# Register components by import side effects.
import src.models.backbones.resnet18  # noqa: F401
import src.models.backbones.convnet  # noqa: F401
import src.models.backbones.convnet_mnist_official  # noqa: F401
import src.models.backbones.vgg16  # noqa: F401
import src.models.backbones.wrn28_10  # noqa: F401
import src.models.heads.edl_head  # noqa: F401
import src.losses.edl_fixed  # noqa: F401
import src.losses.edl_info_adaptive  # noqa: F401
import src.losses.iedl_ref  # noqa: F401
import src.scores.vacuity  # noqa: F401
import src.scores.maxp  # noqa: F401
import src.scores.alpha0  # noqa: F401

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


def _loss_kwargs(cfg) -> Dict[str, Any]:
    loss_name = str(cfg.loss.name)
    if loss_name == "edl_fixed":
        return {"lam": cfg.loss.lambda_value, "anneal_epochs": cfg.loss.anneal_epochs}
    if loss_name == "iedl_ref":
        return {
            "lambda_kl": cfg.loss.lambda_kl,
            "fisher_c": cfg.loss.fisher_c,
            "kl_anneal_epochs": cfg.loss.kl_anneal_epochs,
            "lambda_logdet": cfg.loss.lambda_logdet,
        }
    if loss_name == "info_edl":
        return {
            "beta": cfg.loss.beta,
            "gamma": cfg.loss.gamma,
            "info_type": cfg.loss.info_type,
            "gate_type": cfg.loss.gate_type,
            "detach_weight": cfg.loss.detach_weight,
        }
    raise ValueError(f"Unsupported loss.name: {loss_name}")


class InfoEDLLightningModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        # Avoid serializing OmegaConf DictConfig into checkpoints (PyTorch 2.6
        # weights_only load blocks unknown globals by default).
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        backbone_cls = BACKBONE_REGISTRY.get(cfg.model.backbone)
        backbone_kwargs = {}
        signature = inspect.signature(backbone_cls.__init__)
        if "pretrained" in signature.parameters:
            backbone_kwargs["pretrained"] = cfg.model.pretrained
        if "checkpoint_path" in signature.parameters:
            backbone_kwargs["checkpoint_path"] = str(getattr(cfg.model, "backbone_checkpoint", ""))
        self.backbone: nn.Module = backbone_cls(**backbone_kwargs)
        assert_module_instance(self.backbone, BackboneProtocol, "backbone")

        head_cls = HEAD_REGISTRY.get(cfg.model.head)
        self.head: nn.Module = head_cls(
            in_dim=self.backbone.out_dim,
            num_classes=cfg.model.num_classes,
            evidence_fn=cfg.model.evidence_fn,
        )
        assert_module_instance(self.head, HeadProtocol, "head")

        loss_cls = LOSS_REGISTRY.get(cfg.loss.name)
        self.loss_fn = loss_cls(**_loss_kwargs(cfg))
        assert_module_instance(self.loss_fn, LossProtocol, "loss")

        self.score_fn = SCORE_REGISTRY.get(cfg.score.name)
        if not callable(self.score_fn):
            raise TypeError("score function must be callable")

        self._shape_checked = False
        self._last_grad_norm = torch.tensor(float("nan"))
        self._epoch_start_time = None

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
        loss_out = self.loss_fn(out["alpha"], y, epoch=float(self.current_epoch))
        pred = out["probs"].argmax(dim=1)
        acc = (pred == y).float().mean()
        uncertainty = out["uncertainty_score"]
        correct_mask = pred == y
        wrong_mask = ~correct_mask
        unc_correct = (
            uncertainty[correct_mask].mean() if correct_mask.any() else torch.tensor(float("nan"), device=x.device)
        )
        unc_wrong = (
            uncertainty[wrong_mask].mean() if wrong_mask.any() else torch.tensor(float("nan"), device=x.device)
        )
        evidence_sum = (out["alpha"].sum(dim=1) - self.cfg.model.num_classes).mean()

        lambda_mean = float(loss_out["aux"].get("lambda_mean", float("nan")))
        lambda_std = float(loss_out["aux"].get("lambda_std", float("nan")))
        lambda_min = float(loss_out["aux"].get("lambda_min", float("nan")))
        lambda_max = float(loss_out["aux"].get("lambda_max", float("nan")))
        info = float(loss_out["aux"].get("info", float("nan")))
        info_std = float(loss_out["aux"].get("info_std", float("nan")))
        fisher_trace = float(loss_out["aux"].get("fisher_trace", float("nan")))
        kl_weighted = loss_out["reg"] * lambda_mean

        self.log(f"{stage}/loss", loss_out["total"], prog_bar=(stage != "train"), on_epoch=True)
        self.log(f"{stage}/fit", loss_out["fit"], on_epoch=True)
        self.log(f"{stage}/reg", loss_out["reg"], on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True)
        for key, value in loss_out["aux"].items():
            self.log(f"{stage}/aux/{key}", value, on_epoch=True)

        if stage == "train":
            self.log("Loss/Total", loss_out["total"], on_epoch=True, prog_bar=True)
            self.log("Loss/Risk", loss_out["fit"], on_epoch=True)
            self.log("Loss/KL_raw", loss_out["reg"], on_epoch=True)
            self.log("Loss/KL_weighted", kl_weighted, on_epoch=True)
            self.log("Metric/Info", info, on_epoch=True)
            self.log("Metric/Info_Std", info_std, on_epoch=True)
            self.log("Metric/Fisher_Trace", fisher_trace, on_epoch=True)
            self.log("Metric/Lambda_Mean", lambda_mean, on_epoch=True)
            self.log("Metric/Lambda_Std", lambda_std, on_epoch=True)
            self.log("Metric/Lambda_Min", lambda_min, on_epoch=True)
            self.log("Metric/Lambda_Max", lambda_max, on_epoch=True)
            self.log("Uncertainty/Train_Mean", uncertainty.mean(), on_epoch=True)
            self.log("Uncertainty/Correct", unc_correct, on_epoch=True)
            self.log("Uncertainty/Wrong", unc_wrong, on_epoch=True)
            self.log("Evidence/Total_Sum", evidence_sum, on_epoch=True)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            self.log("val_uncertainty", uncertainty.mean(), on_epoch=True)
            self.log("val_fisher_weight", lambda_mean, on_epoch=True)
        return loss_out["total"]

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._common_step(batch, "test")

    def on_after_backward(self) -> None:
        grad_sq = 0.0
        for param in self.parameters():
            if param.grad is not None:
                grad_sq += float(param.grad.detach().norm(2).item() ** 2)
        self._last_grad_norm = torch.tensor(grad_sq**0.5, device=self.device)

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self) -> None:
        epoch_time = float("nan")
        if self._epoch_start_time is not None:
            epoch_time = time.perf_counter() - self._epoch_start_time
        self.log("System/Epoch_Time_sec", epoch_time, on_epoch=True)
        self.log("System/Gradient_Norm", self._last_grad_norm, on_epoch=True)
        self.log("epoch", float(self.current_epoch + 1), on_epoch=True)
        optim = self.optimizers()
        if isinstance(optim, list):
            optim = optim[0] if optim else None
        if hasattr(optim, "optimizer"):
            optim = optim.optimizer
        if optim is not None:
            self.log("lr", float(optim.param_groups[0]["lr"]), on_epoch=True)

    def predict_scores(self, batch: Any) -> Tensor:
        x, _ = batch
        out = self(x)
        return out["uncertainty_score"]

    def configure_optimizers(self):
        opt_name = str(self.cfg.optimizer.name).lower()
        if opt_name == "adam":
            optim = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif opt_name == "adamw":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        elif opt_name == "sgd":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer.name: {self.cfg.optimizer.name}")

        sched_name = str(self.cfg.scheduler.name).lower()
        if sched_name == "none":
            return {"optimizer": optim}
        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.cfg.trainer.max_epochs)
            return {"optimizer": optim, "lr_scheduler": sched}
        raise ValueError(f"Unsupported scheduler.name: {self.cfg.scheduler.name}")
