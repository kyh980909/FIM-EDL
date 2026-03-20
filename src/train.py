from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.data.datamodule import InfoEDLDataModule
from src.models.lit_module import InfoEDLLightningModule
from src.registry.validators import validate_registry_bindings
from src.reporting.collector import LocalCollector


def _maybe_loss_value(cfg: DictConfig, key: str):
    return OmegaConf.select(cfg, f"loss.{key}", default=None)


def _wandb_tags(cfg: DictConfig) -> list[str]:
    base = list(cfg.logging.wandb.tags)
    exp_tags = list(getattr(cfg.experiment, "wandb_tags", []))
    auto = [f"experiment:{cfg.experiment.name}", f"loss:{cfg.loss.name}"]
    # Keep order but remove duplicates.
    out = []
    for tag in base + exp_tags + auto:
        if tag not in out:
            out.append(tag)
    return out


def _build_logger(cfg: DictConfig):
    if cfg.logging.backend == "wandb":
        logger = WandbLogger(
            project=cfg.logging.wandb.project,
            group=cfg.experiment.name,
            name=f"{cfg.experiment.name}_seed{cfg.seed}",
            save_dir=cfg.logging.local_dir,
            mode=cfg.logging.wandb.mode,
            tags=_wandb_tags(cfg),
        )
        loss_name = str(cfg.loss.name)
        if loss_name == "info_edl":
            method = "info_edl"
        elif loss_name == "iedl_ref":
            method = "iedl_ref"
        else:
            method = "edl"
        logger.experiment.config.update(
            {
                "method": method,
                "method_variant": str(cfg.experiment.method_variant),
                "suite": str(cfg.experiment.suite),
                "epochs": int(cfg.trainer.max_epochs),
                "lr": float(cfg.optimizer.lr),
                "num_classes": int(cfg.model.num_classes),
            },
            allow_val_change=True,
        )
        optional_loss_items = {
            "beta": _maybe_loss_value(cfg, "beta"),
            "gamma": _maybe_loss_value(cfg, "gamma"),
            "lambda_value": _maybe_loss_value(cfg, "lambda_value"),
            "anneal_epochs": _maybe_loss_value(cfg, "anneal_epochs"),
            "lambda_kl": _maybe_loss_value(cfg, "lambda_kl"),
            "lambda_logdet": _maybe_loss_value(cfg, "lambda_logdet"),
            "fisher_c": _maybe_loss_value(cfg, "fisher_c"),
            "kl_anneal_epochs": _maybe_loss_value(cfg, "kl_anneal_epochs"),
            "info_type": _maybe_loss_value(cfg, "info_type"),
            "gate_type": _maybe_loss_value(cfg, "gate_type"),
            "detach_weight": _maybe_loss_value(cfg, "detach_weight"),
            "objective": _maybe_loss_value(cfg, "objective"),
        }
        logger.experiment.config.update(
            {k: v for k, v in optional_loss_items.items() if v is not None},
            allow_val_change=True,
        )
        return logger
    return CSVLogger(save_dir=cfg.logging.local_dir, name=cfg.experiment.name)

def _enable_checkpoint_safe_globals() -> None:
    # Compatibility for checkpoints containing OmegaConf classes.
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


def run_train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _enable_checkpoint_safe_globals()
    validate_registry_bindings(cfg)

    collector = LocalCollector(cfg)
    datamodule = InfoEDLDataModule(cfg)
    model = InfoEDLLightningModule(cfg)

    monitor = str(cfg.trainer.early_stopping_monitor) if bool(cfg.trainer.early_stopping) else "val/acc"
    mode = str(cfg.trainer.early_stopping_mode) if bool(cfg.trainer.early_stopping) else "max"
    ckpt = ModelCheckpoint(
        dirpath=str(Path(collector.run_dir) / "checkpoints"),
        save_top_k=1,
        monitor=monitor,
        mode=mode,
        filename="best",
    )
    callbacks = [ckpt]
    if bool(cfg.trainer.early_stopping):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.trainer.early_stopping_monitor),
                mode=str(cfg.trainer.early_stopping_mode),
                patience=int(cfg.trainer.early_stopping_patience),
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices) if isinstance(cfg.trainer.devices, int) or str(cfg.trainer.devices).isdigit() else cfg.trainer.devices,
        precision=str(cfg.trainer.precision),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        limit_train_batches=OmegaConf.select(cfg, "trainer.limit_train_batches", default=1.0),
        limit_val_batches=OmegaConf.select(cfg, "trainer.limit_val_batches", default=1.0),
        logger=_build_logger(cfg),
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    metrics = test_metrics[0] if test_metrics else {}
    collector.append_metric(
        method=cfg.experiment.name,
        seed=cfg.seed,
        dataset=cfg.data.id,
        split="test",
        metrics={"accuracy": float(metrics.get("test/acc", 0.0))},
        method_variant=str(cfg.experiment.method_variant),
        score_type=str(cfg.score.name),
        calibration_type=str(cfg.eval.calibration),
    )
    collector.write_summary(
        {
            "best_model_path": ckpt.best_model_path,
            "seed": cfg.seed,
            "experiment": cfg.experiment.name,
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        }
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_train(cfg)


if __name__ == "__main__":
    main()
