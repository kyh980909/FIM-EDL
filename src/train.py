from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.data.datamodule import InfoEDLDataModule
from src.models.lit_module import InfoEDLLightningModule
from src.registry.validators import validate_registry_bindings
from src.reporting.collector import LocalCollector


def _build_logger(cfg: DictConfig):
    if cfg.logging.backend == "wandb":
        logger = WandbLogger(
            project=cfg.logging.wandb.project,
            group=cfg.experiment.name,
            name=f"{cfg.experiment.name}_seed{cfg.seed}",
            save_dir=cfg.logging.local_dir,
            mode=cfg.logging.wandb.mode,
            tags=list(cfg.logging.wandb.tags),
        )
        method = "fisher" if str(cfg.loss.name) == "info_edl" else "edl"
        logger.experiment.config.update(
            {
                "method": method,
                "epochs": int(cfg.trainer.max_epochs),
                "lr": float(cfg.optimizer.lr),
                "beta": float(cfg.loss.beta),
                "gamma": float(cfg.loss.gamma),
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
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
