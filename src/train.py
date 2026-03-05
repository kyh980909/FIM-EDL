from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.data.datamodule import InfoEDLDataModule
from src.models.lit_module import InfoEDLLightningModule
from src.registry.validators import validate_registry_bindings
from src.reporting.collector import LocalCollector


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
                "beta": float(cfg.loss.beta),
                "gamma": float(cfg.loss.gamma),
                "lambda_kl": float(cfg.loss.lambda_kl),
                "lambda_logdet": float(cfg.loss.lambda_logdet),
                "num_classes": int(cfg.model.num_classes),
                "info_type": str(cfg.loss.info_type),
                "gate_type": str(cfg.loss.gate_type),
                "detach_weight": bool(cfg.loss.detach_weight),
                "objective": str(cfg.loss.objective),
            },
            allow_val_change=True,
        )
        return logger
    return CSVLogger(save_dir=cfg.logging.local_dir, name=cfg.experiment.name)

def _enable_checkpoint_safe_globals() -> None:
    # Compatibility for checkpoints containing OmegaConf classes.
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _enable_checkpoint_safe_globals()
    validate_registry_bindings(cfg)

    collector = LocalCollector(cfg)
    datamodule = InfoEDLDataModule(cfg)
    model = InfoEDLLightningModule(cfg)

    ckpt = ModelCheckpoint(
        dirpath=str(Path(collector.run_dir) / "checkpoints"),
        save_top_k=1,
        monitor="val/acc",
        mode="max",
        filename="best",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=_build_logger(cfg),
        callbacks=[ckpt],
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


if __name__ == "__main__":
    main()
