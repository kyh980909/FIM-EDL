from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.data.datamodule import InfoEDLDataModule
from src.metrics.ood_metrics import auroc_and_fpr95
from src.models.lit_module import InfoEDLLightningModule
from src.reporting.collector import LocalCollector


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

    for name, loader in datamodule.ood_dataloaders().items():
        ood_scores, _ = _collect_scores(model, loader)
        auroc, fpr95, meta = auroc_and_fpr95(id_scores=id_scores, ood_scores=ood_scores)
        collector.append_metric(
            method=cfg.experiment.name,
            seed=cfg.seed,
            dataset=name,
            split="eval",
            metrics={"accuracy": accuracy, "auroc": auroc, "fpr95": fpr95},
            extra={"threshold_meta": meta},
        )

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
