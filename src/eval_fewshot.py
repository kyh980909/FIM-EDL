from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig
from sklearn.metrics import average_precision_score
from torch.utils.data import ConcatDataset, DataLoader

from src.data.adapters.miniimagenet_adapter import MiniImageNetAdapter
from src.metrics.ood_metrics import aupr, auroc_and_fpr95
from src.models.lit_module import InfoEDLLightningModule
from src.reporting.collector import LocalCollector

# Register components by import side effects.
import src.losses.edl_fixed  # noqa: F401
import src.losses.edl_info_adaptive  # noqa: F401
import src.losses.iedl_ref  # noqa: F401
import src.models.heads.edl_head  # noqa: F401

from src.registry.heads import HEAD_REGISTRY
from src.registry.losses import LOSS_REGISTRY


def _enable_checkpoint_safe_globals() -> None:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


def _loss_kwargs(cfg) -> Dict[str, float]:
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
        }
    raise ValueError(f"Unsupported loss.name: {loss_name}")


def _build_eval_dataset(adapter: MiniImageNetAdapter, split_name: str):
    split = str(split_name).lower()
    if split in {"train", "val", "test"}:
        return adapter.imagefolder_dataset(f"miniimagenet/{split}", train=False)
    if split == "valtest":
        return ConcatDataset(
            [
                adapter.imagefolder_dataset("miniimagenet/val", train=False),
                adapter.imagefolder_dataset("miniimagenet/test", train=False),
            ]
        )
    if split == "all":
        return ConcatDataset(
            [
                adapter.imagefolder_dataset("miniimagenet/train", train=False),
                adapter.imagefolder_dataset("miniimagenet/val", train=False),
                adapter.imagefolder_dataset("miniimagenet/test", train=False),
            ]
        )
    raise ValueError(f"Unsupported fewshot.eval_split: {split_name}")


def _split_class_sets(root: Path) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for split in ["train", "val", "test"]:
        split_root = root / "miniimagenet" / split
        out[split] = {p.name for p in split_root.iterdir() if p.is_dir()} if split_root.exists() else set()
    return out


def _validate_paper_protocol(cfg: DictConfig, adapter: MiniImageNetAdapter) -> None:
    if not bool(cfg.fewshot.paper_protocol_strict):
        return
    if str(cfg.model.backbone) != "wrn28_10":
        raise ValueError("Paper few-shot protocol requires model.backbone=wrn28_10")
    if not str(getattr(cfg.model, "backbone_checkpoint", "")):
        raise ValueError(
            "Paper few-shot protocol requires model.backbone_checkpoint to point to the external pretrained WRN-28-10 weights."
        )
    if str(cfg.fewshot.eval_split) != "test":
        raise ValueError("Paper few-shot protocol requires fewshot.eval_split=test")
    if int(cfg.fewshot.query_per_class) not in {-1, min(15, int(cfg.fewshot.shot))}:
        raise ValueError("Paper few-shot protocol uses query_per_class=min(15, shot)")

    split_classes = _split_class_sets(adapter.root)
    train_classes = split_classes["train"]
    val_classes = split_classes["val"]
    test_classes = split_classes["test"]
    if train_classes & val_classes or train_classes & test_classes or val_classes & test_classes:
        raise ValueError(
            "Paper mini-ImageNet protocol requires disjoint meta-train/meta-val/meta-test classes, but the current folders share class names."
        )


def _extract_features(
    backbone: torch.nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).detach().cpu())
            labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _build_index(labels: torch.Tensor) -> Dict[int, np.ndarray]:
    out: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        out[int(label)].append(idx)
    return {label: np.asarray(indices, dtype=np.int64) for label, indices in out.items()}


def _score_map(alpha: torch.Tensor, probs: torch.Tensor) -> Dict[str, np.ndarray]:
    alpha_np = alpha.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    alpha0 = alpha_np.sum(axis=1)
    maxp = probs_np.max(axis=1)
    maxalpha = alpha_np.max(axis=1)
    vacuity = alpha_np.shape[1] / np.clip(alpha0, 1e-12, None)
    return {"maxp": maxp, "maxalpha": maxalpha, "alpha0": alpha0, "vacuity": vacuity}


def _ood_score_from_raw(score_name: str, raw: np.ndarray) -> np.ndarray:
    if score_name in {"maxp", "alpha0", "maxalpha"}:
        return -raw
    return raw


def _confidence_aupr(conf_scores: np.ndarray, correct: np.ndarray) -> float:
    labels = correct.astype(np.int32)
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, conf_scores))


def _episode_query_count(cfg: DictConfig, sampled_classes: Iterable[int], class_to_indices: Dict[int, np.ndarray]) -> int:
    requested = int(cfg.fewshot.query_per_class)
    if requested > 0:
        return requested
    desired = min(15, int(cfg.fewshot.shot))
    available = min(int(class_to_indices[label].shape[0]) - int(cfg.fewshot.shot) for label in sampled_classes)
    if available <= 0:
        raise ValueError("Not enough examples per class for few-shot support/query split")
    if bool(cfg.fewshot.paper_protocol_strict) and available < desired:
        raise ValueError(
            f"Paper few-shot protocol requires {desired} query samples per class, but only {available} are available."
        )
    return max(1, min(desired, available))


def _adapt_episode_head(
    cfg: DictConfig,
    support_feats: torch.Tensor,
    support_labels: torch.Tensor,
    device: torch.device,
) -> torch.nn.Module:
    head_cls = HEAD_REGISTRY.get(cfg.model.head)
    head = head_cls(
        in_dim=int(support_feats.shape[1]),
        num_classes=int(torch.unique(support_labels).numel()),
        evidence_fn=str(cfg.model.evidence_fn),
    ).to(device)
    loss_cls = LOSS_REGISTRY.get(cfg.loss.name)
    loss_fn = loss_cls(**_loss_kwargs(cfg))
    batch_size = int(cfg.fewshot.adapt_batch_size)
    if batch_size <= 0:
        batch_size = int(support_feats.shape[0])

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg.fewshot.adapt_lr),
        weight_decay=float(cfg.fewshot.adapt_weight_decay),
    )
    support_feats = support_feats.to(device)
    support_labels = support_labels.to(device)

    head.train()
    for step in range(int(cfg.fewshot.adapt_steps)):
        perm = torch.randperm(support_feats.shape[0], device=device)
        for start in range(0, support_feats.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            out = head(support_feats[idx])
            loss = loss_fn(out["alpha"], support_labels[idx], epoch=float(step))["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    head.eval()
    return head


def _mean_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "std": float("nan"), "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    ci95 = float(1.96 * std / max(arr.size, 1) ** 0.5)
    return {"mean": mean, "ci95": ci95, "std": std, "n": int(arr.size)}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not bool(cfg.fewshot.enabled):
        raise ValueError("fewshot.enabled must be true for src.eval_fewshot")
    if str(cfg.data.id).lower() not in {"miniimagenet", "mini-imagenet"}:
        raise ValueError("Few-shot episodic evaluation currently supports data.id=miniimagenet only")
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be provided for few-shot eval")

    pl.seed_everything(cfg.seed, workers=True)
    _enable_checkpoint_safe_globals()
    collector = LocalCollector(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.trainer.accelerator != "cpu" else "cpu")
    model = InfoEDLLightningModule.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    model.to(device)
    model.eval()

    adapter = MiniImageNetAdapter(root=str(cfg.data.root))
    _validate_paper_protocol(cfg, adapter)
    id_dataset = _build_eval_dataset(adapter, str(cfg.fewshot.eval_split))
    ood_dataset = adapter.imagefolder_dataset(f"{cfg.fewshot.ood_dataset}/test", train=False)

    feature_batch_size = min(int(cfg.data.batch_size), 64)
    id_feats, id_labels = _extract_features(
        backbone=model.backbone,
        dataset=id_dataset,
        batch_size=feature_batch_size,
        num_workers=int(cfg.data.num_workers),
        device=device,
    )
    ood_feats, _ = _extract_features(
        backbone=model.backbone,
        dataset=ood_dataset,
        batch_size=feature_batch_size,
        num_workers=int(cfg.data.num_workers),
        device=device,
    )

    class_to_indices = _build_index(id_labels)
    available_classes = sorted(class_to_indices.keys())
    if len(available_classes) < int(cfg.fewshot.way):
        raise ValueError("Not enough classes for requested few-shot way")

    rng = np.random.default_rng(int(cfg.seed))
    accuracy_vals: List[float] = []
    conf_vals: Dict[str, List[float]] = {name: [] for name in list(cfg.eval.confidence_scores)}
    ood_aupr_vals: Dict[str, List[float]] = {name: [] for name in list(cfg.eval.scores)}
    ood_auroc_vals: Dict[str, List[float]] = {name: [] for name in list(cfg.eval.scores)}
    ood_fpr95_vals: Dict[str, List[float]] = {name: [] for name in list(cfg.eval.scores)}
    query_counts: List[int] = []

    for _ in range(int(cfg.fewshot.episodes)):
        sampled = rng.choice(available_classes, size=int(cfg.fewshot.way), replace=False).tolist()
        query_per_class = _episode_query_count(cfg, sampled, class_to_indices)
        query_counts.append(query_per_class)

        support_idx: List[int] = []
        support_labels: List[int] = []
        query_idx: List[int] = []
        query_labels: List[int] = []
        for new_label, cls in enumerate(sampled):
            picks = rng.choice(class_to_indices[cls], size=int(cfg.fewshot.shot) + query_per_class, replace=False)
            support_idx.extend(picks[: int(cfg.fewshot.shot)].tolist())
            support_labels.extend([new_label] * int(cfg.fewshot.shot))
            query_idx.extend(picks[int(cfg.fewshot.shot) :].tolist())
            query_labels.extend([new_label] * query_per_class)

        ood_idx = rng.choice(ood_feats.shape[0], size=len(query_idx), replace=False)

        support_feat_t = id_feats[support_idx]
        support_label_t = torch.tensor(support_labels, dtype=torch.long)
        query_feat_t = id_feats[query_idx].to(device)
        query_label_t = torch.tensor(query_labels, dtype=torch.long, device=device)
        ood_feat_t = ood_feats[ood_idx].to(device)

        head = _adapt_episode_head(cfg, support_feat_t, support_label_t, device=device)
        with torch.no_grad():
            id_out = head(query_feat_t)
            ood_out = head(ood_feat_t)

        pred = id_out["probs"].argmax(dim=1)
        correct = (pred == query_label_t).detach().cpu().numpy().astype(np.float32)
        accuracy_vals.append(float(correct.mean()))

        id_scores = _score_map(alpha=id_out["alpha"], probs=id_out["probs"])
        ood_scores = _score_map(alpha=ood_out["alpha"], probs=ood_out["probs"])

        for score_name in conf_vals:
            if score_name not in id_scores:
                continue
            conf_vals[score_name].append(_confidence_aupr(id_scores[score_name], correct))

        for score_name in ood_aupr_vals:
            if score_name not in id_scores or score_name not in ood_scores:
                continue
            id_score = _ood_score_from_raw(score_name, id_scores[score_name])
            ood_score = _ood_score_from_raw(score_name, ood_scores[score_name])
            auroc_v, fpr95_v, _ = auroc_and_fpr95(id_scores=id_score, ood_scores=ood_score)
            aupr_v = aupr(id_scores=id_score, ood_scores=ood_score)
            ood_aupr_vals[score_name].append(aupr_v)
            ood_auroc_vals[score_name].append(auroc_v)
            ood_fpr95_vals[score_name].append(fpr95_v)

    accuracy_summary = _mean_ci95(accuracy_vals)
    conf_summary = {name: _mean_ci95(vals) for name, vals in conf_vals.items()}
    ood_aupr_summary = {name: _mean_ci95(vals) for name, vals in ood_aupr_vals.items()}
    ood_auroc_summary = {name: _mean_ci95(vals) for name, vals in ood_auroc_vals.items()}
    ood_fpr95_summary = {name: _mean_ci95(vals) for name, vals in ood_fpr95_vals.items()}

    for score_name, stats in conf_summary.items():
        collector.append_metric(
            method=cfg.experiment.name,
            seed=int(cfg.seed),
            dataset=str(cfg.data.id),
            split="fewshot_eval",
            metrics={
                "accuracy": accuracy_summary["mean"],
                "aupr": stats["mean"],
                "auroc": float("nan"),
                "fpr95": float("nan"),
            },
            method_variant=str(cfg.experiment.method_variant),
            score_type=score_name,
            calibration_type=str(cfg.eval.calibration),
            extra={
                "metric_family": "confidence",
                "accuracy_ci95": accuracy_summary["ci95"],
                "aupr_ci95": stats["ci95"],
                "episodes": int(cfg.fewshot.episodes),
                "way": int(cfg.fewshot.way),
                "shot": int(cfg.fewshot.shot),
                "query_per_class_mean": float(np.mean(query_counts)),
                "eval_split": str(cfg.fewshot.eval_split),
            },
        )

    for score_name, stats in ood_aupr_summary.items():
        collector.append_metric(
            method=cfg.experiment.name,
            seed=int(cfg.seed),
            dataset=str(cfg.fewshot.ood_dataset),
            split="fewshot_eval",
            metrics={
                "accuracy": accuracy_summary["mean"],
                "aupr": stats["mean"],
                "auroc": ood_auroc_summary[score_name]["mean"],
                "fpr95": ood_fpr95_summary[score_name]["mean"],
            },
            method_variant=str(cfg.experiment.method_variant),
            score_type=score_name,
            calibration_type=str(cfg.eval.calibration),
            extra={
                "metric_family": "ood",
                "accuracy_ci95": accuracy_summary["ci95"],
                "aupr_ci95": stats["ci95"],
                "auroc_ci95": ood_auroc_summary[score_name]["ci95"],
                "fpr95_ci95": ood_fpr95_summary[score_name]["ci95"],
                "episodes": int(cfg.fewshot.episodes),
                "way": int(cfg.fewshot.way),
                "shot": int(cfg.fewshot.shot),
                "query_per_class_mean": float(np.mean(query_counts)),
                "eval_split": str(cfg.fewshot.eval_split),
            },
        )

    collector.write_summary(
        {
            "checkpoint": str(Path(cfg.checkpoint).resolve()),
            "experiment": str(cfg.experiment.name),
            "seed": int(cfg.seed),
            "fewshot": {
                "episodes": int(cfg.fewshot.episodes),
                "way": int(cfg.fewshot.way),
                "shot": int(cfg.fewshot.shot),
                "eval_split": str(cfg.fewshot.eval_split),
                "query_per_class_mean": float(np.mean(query_counts)),
                "adapt_steps": int(cfg.fewshot.adapt_steps),
                "adapt_lr": float(cfg.fewshot.adapt_lr),
            },
            "accuracy": accuracy_summary,
            "confidence_aupr": conf_summary,
            "ood_aupr": ood_aupr_summary,
            "ood_auroc": ood_auroc_summary,
            "ood_fpr95": ood_fpr95_summary,
        }
    )


if __name__ == "__main__":
    main()
