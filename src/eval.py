from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig

from src.data.datamodule import InfoEDLDataModule
from src.metrics.ood_metrics import (
    aupr,
    aurc_from_confidence,
    auroc_and_fpr95,
    multiclass_ece,
    multiclass_nll,
)
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


def _temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("eval.temperature must be > 0")
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / float(temperature)
    t = torch.from_numpy(scaled).float()
    p = F.softmax(t, dim=1).numpy()
    return p


def _collect_outputs(model: InfoEDLLightningModule, loader, temperature: float) -> Dict[str, np.ndarray]:
    model.eval()
    all_alpha: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(model.device))
            all_alpha.append(out["alpha"].detach().cpu().numpy())
            all_probs.append(out["probs"].detach().cpu().numpy())
            all_labels.append(y.numpy())
    alpha = np.concatenate(all_alpha)
    probs = np.concatenate(all_probs)
    probs = _temperature_scale_probs(probs, temperature=temperature)
    labels = np.concatenate(all_labels)
    return {"alpha": alpha, "probs": probs, "labels": labels}


def _score_map(alpha: np.ndarray, probs: np.ndarray) -> Dict[str, np.ndarray]:
    alpha0 = alpha.sum(axis=1)
    maxp = probs.max(axis=1)
    vacuity = alpha.shape[1] / np.clip(alpha0, 1e-12, None)
    return {"maxp": maxp, "alpha0": alpha0, "vacuity": vacuity}


def _ood_score_from_raw(score_name: str, raw: np.ndarray) -> np.ndarray:
    # Higher values should indicate more likely OOD for AUROC/AUPR.
    if score_name in {"maxp", "alpha0"}:
        return -raw
    return raw


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

    temperature = float(cfg.eval.temperature if cfg.eval.calibration == "temperature" else 1.0)
    id_out = _collect_outputs(model, datamodule.test_dataloader(), temperature=temperature)
    id_scores_raw = _score_map(alpha=id_out["alpha"], probs=id_out["probs"])

    id_pred = id_out["probs"].argmax(axis=1)
    id_correct = (id_pred == id_out["labels"]).astype(np.float32)
    accuracy = float(id_correct.mean())
    nll = multiclass_nll(id_out["probs"], id_out["labels"])
    ece = multiclass_ece(id_out["probs"], id_out["labels"])
    aurc = aurc_from_confidence(id_out["probs"].max(axis=1), id_correct)

    rows: List[Dict[str, float]] = []
    for name, loader in datamodule.ood_dataloaders().items():
        ood_out = _collect_outputs(model, loader, temperature=temperature)
        ood_scores_raw = _score_map(alpha=ood_out["alpha"], probs=ood_out["probs"])

        for score_name in list(cfg.eval.scores):
            id_score = _ood_score_from_raw(score_name, id_scores_raw[score_name])
            ood_score = _ood_score_from_raw(score_name, ood_scores_raw[score_name])
            auroc, fpr95, meta = auroc_and_fpr95(id_scores=id_score, ood_scores=ood_score)
            aupr_v = aupr(id_scores=id_score, ood_scores=ood_score)

            row = {
                "dataset": name,
                "score_type": score_name,
                "acc": accuracy,
                "nll": nll,
                "ece": ece,
                "aurc": aurc,
                "auroc": auroc,
                "aupr": aupr_v,
                "fpr95": fpr95,
            }
            rows.append(row)
            collector.append_metric(
                method=cfg.experiment.name,
                seed=cfg.seed,
                dataset=name,
                split="eval",
                metrics={
                    "accuracy": accuracy,
                    "nll": nll,
                    "ece": ece,
                    "aurc": aurc,
                    "auroc": auroc,
                    "aupr": aupr_v,
                    "fpr95": fpr95,
                },
                method_variant=str(cfg.experiment.method_variant),
                score_type=score_name,
                calibration_type=str(cfg.eval.calibration),
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
                "calibration": str(cfg.eval.calibration),
                "temperature": float(temperature),
            },
            mode=cfg.logging.wandb.mode,
            tags=_wandb_tags(cfg),
            reinit=True,
        )
        for row in rows:
            prefix = f"eval/{row['dataset']}/{row['score_type']}"
            run.log(
                {
                    f"{prefix}/acc": row["acc"],
                    f"{prefix}/nll": row["nll"],
                    f"{prefix}/ece": row["ece"],
                    f"{prefix}/auroc": row["auroc"],
                    f"{prefix}/aupr": row["aupr"],
                    f"{prefix}/fpr95": row["fpr95"],
                    f"{prefix}/aurc": row["aurc"],
                }
            )
        run.finish()

    collector.write_summary(
        {
            "checkpoint": str(Path(ckpt_path).resolve()),
            "seed": cfg.seed,
            "experiment": cfg.experiment.name,
            "id_accuracy": accuracy,
            "calibration": str(cfg.eval.calibration),
            "temperature": temperature,
            "scores": list(cfg.eval.scores),
        }
    )


if __name__ == "__main__":
    main()
