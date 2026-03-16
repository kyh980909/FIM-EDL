from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.data.datamodule import InfoEDLDataModule
from src.models.lit_module import InfoEDLLightningModule


METHOD_DISPLAY = {
    "iedl_ref": "I-EDL Ref",
    "info_edl": "Info-EDL",
}

OOD_BY_DATASET = {
    "mnist": ["kmnist", "fmnist"],
    "cifar10": ["svhn", "cifar100"],
}

SCORE_SPECS = [("maxp", "Max.P"), ("alpha0", r"$\alpha_0$")]


def _enable_checkpoint_safe_globals() -> None:
    if hasattr(torch.serialization, "add_safe_globals"):
        from omegaconf.dictconfig import DictConfig as OmegaDictConfig
        from omegaconf.listconfig import ListConfig as OmegaListConfig

        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


def _latest_run_dir(method: str, dataset: str, seed: int) -> Path:
    seed_dir = Path("runs") / method / f"seed_{seed}"
    latest: Path | None = None
    for run_dir in sorted(seed_dir.iterdir()):
        cfg_path = run_dir / "config_resolved.yaml"
        ckpt_path = run_dir / "checkpoints" / "best.ckpt"
        if not cfg_path.exists() or not ckpt_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if str(cfg.get("data", {}).get("id")) != dataset:
            continue
        latest = run_dir
    if latest is None:
        raise FileNotFoundError(f"No run found for method={method}, dataset={dataset}, seed={seed}")
    return latest


def _collect_outputs(model: InfoEDLLightningModule, loader) -> Dict[str, np.ndarray]:
    model.eval()
    all_alpha: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for x, _y in loader:
            out = model(x.to(model.device))
            all_alpha.append(out["alpha"].detach().cpu().numpy())
            all_probs.append(out["probs"].detach().cpu().numpy())
    alpha = np.concatenate(all_alpha)
    probs = np.concatenate(all_probs)
    return {"alpha": alpha, "probs": probs}


def _score_map(alpha: np.ndarray, probs: np.ndarray) -> Dict[str, np.ndarray]:
    alpha0 = alpha.sum(axis=1)
    maxp = probs.max(axis=1)
    return {"maxp": maxp, "alpha0": alpha0}


def _load_cfg(run_dir: Path):
    cfg = OmegaConf.create(yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8")))
    data = cfg.data
    if "normalize" not in data:
        data.normalize = True
    if "random_rotation_degrees" not in data:
        data.random_rotation_degrees = 0.0
    if "val_use_train_transform" not in data:
        data.val_use_train_transform = False
    if "image_size" not in data:
        data.image_size = 32 if str(data.id) == "cifar10" else 32
    if "grayscale_to_rgb" not in data:
        data.grayscale_to_rgb = str(data.id) == "mnist"
    if "random_crop_padding" not in data:
        data.random_crop_padding = 2 if str(data.id) == "mnist" else 0
    return cfg


def _plot_dataset_histograms(dataset: str, methods: List[str], seed: int, out_path: Path) -> None:
    ood_names = OOD_BY_DATASET[dataset]
    fig, axes = plt.subplots(
        len(methods),
        len(ood_names) * len(SCORE_SPECS),
        figsize=(4.2 * len(ood_names) * len(SCORE_SPECS), 3.2 * len(methods)),
        dpi=180,
        squeeze=False,
    )

    for row_idx, method in enumerate(methods):
        try:
            run_dir = _latest_run_dir(method=method, dataset=dataset, seed=seed)
            cfg = _load_cfg(run_dir)
            cfg.checkpoint = str((run_dir / "checkpoints" / "best.ckpt").resolve())
            cfg.logging.backend = "csv"
            cfg.data.ood_list = ood_names
            cfg.data.num_workers = 0

            datamodule = InfoEDLDataModule(cfg)
            datamodule.setup(stage="predict")

            _enable_checkpoint_safe_globals()
            model = InfoEDLLightningModule.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
            device = "cuda" if torch.cuda.is_available() and str(cfg.trainer.accelerator) != "cpu" else "cpu"
            model.to(device)

            id_out = _collect_outputs(model, datamodule.test_dataloader())
            id_scores = _score_map(id_out["alpha"], id_out["probs"])
        except Exception as exc:
            for col_idx in range(len(ood_names) * len(SCORE_SPECS)):
                ax = axes[row_idx][col_idx]
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"{METHOD_DISPLAY.get(method, method)}\nnot available\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            continue

        for ood_idx, ood_name in enumerate(ood_names):
            ood_out = _collect_outputs(model, datamodule.ood_dataloaders()[ood_name])
            ood_scores = _score_map(ood_out["alpha"], ood_out["probs"])
            for score_idx, (score_name, score_label) in enumerate(SCORE_SPECS):
                ax = axes[row_idx][ood_idx * len(SCORE_SPECS) + score_idx]
                id_vals = id_scores[score_name]
                ood_vals = ood_scores[score_name]
                bins = 40
                lo = min(float(id_vals.min()), float(ood_vals.min()))
                hi = max(float(id_vals.max()), float(ood_vals.max()))
                ax.hist(id_vals, bins=bins, range=(lo, hi), density=True, alpha=0.55, color="#2a6f97", label="ID")
                ax.hist(ood_vals, bins=bins, range=(lo, hi), density=True, alpha=0.55, color="#c65d3b", label="OOD")
                if row_idx == 0:
                    ax.set_title(f"{ood_name.upper()} / {score_label}")
                if score_idx == 0:
                    ax.set_ylabel(METHOD_DISPLAY.get(method, method))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.18)
                if row_idx == 0 and ood_idx == 0 and score_idx == 0:
                    ax.legend(frameon=False)

    fig.suptitle(f"ID vs OOD Score Distributions ({dataset.upper()}, seed {seed})", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default="iedl_ref,info_edl")
    parser.add_argument("--datasets", default="mnist,cifar10")
    parser.add_argument("--out-dir", default="results/paper_tables/iedl_histograms")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    out_dir = Path(args.out_dir)

    for dataset in datasets:
        _plot_dataset_histograms(
            dataset=dataset,
            methods=methods,
            seed=int(args.seed),
            out_path=out_dir / f"{dataset}_ood_histograms_seed{args.seed}.png",
        )


if __name__ == "__main__":
    main()
