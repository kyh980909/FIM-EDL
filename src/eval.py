from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig

from src.data.datamodule import InfoEDLDataModule
from src.metrics.ood_metrics import auroc_and_fpr95
from src.models.lit_module import InfoEDLLightningModule
from src.reporting.collector import LocalCollector
from src.reporting.wandb_import import import_wandb


def _wandb_tags(cfg: DictConfig) -> list[str]:
    base = list(cfg.logging.wandb.tags)
    exp_tags = list(getattr(cfg.experiment, "wandb_tags", []))
    auto = [f"experiment:{cfg.experiment.name}", f"loss:{cfg.loss.name}", "eval"]
    out = []
    for tag in base + exp_tags + auto:
        if tag not in out:
            out.append(tag)
    return out


def _enable_checkpoint_safe_globals() -> None:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


def _collect_scores(model: InfoEDLLightningModule, loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(model.device))
            s = out["uncertainty_score"].detach().cpu().numpy()
            all_scores.append(s)
            all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _enable_checkpoint_safe_globals()
    collector = LocalCollector(cfg)

    datamodule = InfoEDLDataModule(cfg)
    datamodule.setup()

    ckpt_path = cfg.checkpoint
    if not ckpt_path:
        raise ValueError("checkpoint must be provided for eval")

    model = InfoEDLLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    device = "cuda" if torch.cuda.is_available() and cfg.trainer.accelerator != "cpu" else "cpu"
    model.to(device)

    id_scores, id_labels = _collect_scores(model, datamodule.test_dataloader())

    pred_correct = []
    with torch.no_grad():
        for x, y in datamodule.test_dataloader():
            out = model(x.to(model.device))
            pred = out["probs"].argmax(dim=1).detach().cpu().numpy()
            pred_correct.append((pred == y.numpy()).astype(np.float32))
    accuracy = float(np.concatenate(pred_correct).mean())

    rows: List[Dict[str, float]] = []
    for name, loader in datamodule.ood_dataloaders().items():
        ood_scores, _ = _collect_scores(model, loader)
        auroc, fpr95, meta = auroc_and_fpr95(id_scores=id_scores, ood_scores=ood_scores)
        row = {"dataset": name, "acc": accuracy, "auroc": auroc, "fpr95": fpr95}
        rows.append(row)
        collector.append_metric(
            method=cfg.experiment.name,
            seed=cfg.seed,
            dataset=name,
            split="eval",
            metrics={"accuracy": accuracy, "auroc": auroc, "fpr95": fpr95},
            extra={"threshold_meta": meta},
        )

    if cfg.logging.backend == "wandb":
        wandb = import_wandb(".")
        run = wandb.init(
            project=cfg.logging.wandb.project,
            name=f"{cfg.experiment.name}_seed{cfg.seed}_eval",
            config={
                "ckpt": str(Path(ckpt_path).resolve()),
                "seed": int(cfg.seed),
                "id_dataset": str(cfg.data.id),
                "ood_datasets": ",".join(cfg.data.ood_list),
                "score_type": str(cfg.score.name),
                "calibration": "none",
            },
            mode=cfg.logging.wandb.mode,
            tags=_wandb_tags(cfg),
            reinit=True,
        )
        for row in rows:
            prefix = f"eval/{row['dataset']}"
            run.log(
                {
                    f"{prefix}/acc": row["acc"],
                    f"{prefix}/nll": float("nan"),
                    f"{prefix}/ece": float("nan"),
                    f"{prefix}/auroc": row["auroc"],
                    f"{prefix}/aupr": float("nan"),
                    f"{prefix}/fpr95": row["fpr95"],
                    f"{prefix}/aurc": float("nan"),
                }
            )
        run.finish()

    collector.write_summary(
        {
            "checkpoint": str(Path(ckpt_path).resolve()),
            "seed": cfg.seed,
            "experiment": cfg.experiment.name,
            "id_accuracy": accuracy,
        }
    )


if __name__ == "__main__":
    main()
