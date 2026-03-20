from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

from src.eval import run_eval
from src.train import run_train


ROOT = Path(__file__).resolve().parents[2]


def _load_cfg(experiment_name: str):
    base = OmegaConf.load(ROOT / "configs" / "config.yaml")
    exp = OmegaConf.load(ROOT / "configs" / "experiment" / f"{experiment_name}.yaml")
    return OmegaConf.merge(base, exp)


def _latest_summary(experiment_name: str, seed: int) -> Path:
    candidates = sorted((ROOT / "runs" / experiment_name / f"seed_{seed}").glob("*/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary.json for {experiment_name} seed {seed}")
    return candidates[-1]


def _checkpoint_from_summary(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["summary"]["best_model_path"])


def _common_mnist_cfg(experiment_name: str, seed: int, suite: str):
    cfg = _load_cfg(experiment_name)
    cfg.seed = seed
    cfg.experiment.suite = suite
    cfg.model.backbone = "convnet"
    cfg.data.id = "mnist"
    cfg.data.ood_list = ["kmnist", "fmnist"]
    cfg.model.num_classes = 10
    cfg.trainer.max_epochs = 200
    cfg.data.batch_size = 64
    cfg.data.val_from_train = True
    cfg.data.val_split = 0.2
    cfg.optimizer.lr = 0.001
    cfg.trainer.precision = "32-true"
    cfg.trainer.accelerator = "cuda"
    cfg.trainer.devices = 1
    cfg.trainer.early_stopping = True
    cfg.trainer.early_stopping_monitor = "val/loss"
    cfg.trainer.early_stopping_mode = "min"
    cfg.trainer.early_stopping_patience = 20
    cfg.logging.backend = "csv"
    cfg.data.num_workers = 0
    return cfg


def _train_and_eval(cfg) -> None:
    run_train(cfg)
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.checkpoint = _checkpoint_from_summary(_latest_summary(str(cfg.experiment.name), int(cfg.seed)))
    run_eval(eval_cfg)


def _run_controller(seed: int, suite: str) -> None:
    variants = [
        ("mnist_fim_detach", "fisher", "exp", True, 1.0, 1.0),
        ("mnist_fim_nodetach", "fisher", "exp", False, 1.0, 1.0),
        ("mnist_alpha0_gate", "alpha0", "exp", True, 1.0, 1.0),
        ("mnist_constant_gate", "fisher", "constant", True, 1.0, 1.0),
    ]
    for method_variant, info_type, gate_type, detach_weight, beta, gamma in variants:
        cfg = _common_mnist_cfg("info_edl", seed=seed, suite=suite)
        cfg.experiment.method_variant = method_variant
        cfg.loss.info_type = info_type
        cfg.loss.gate_type = gate_type
        cfg.loss.detach_weight = detach_weight
        cfg.loss.beta = beta
        cfg.loss.gamma = gamma
        _train_and_eval(cfg)


def _run_fixed(seed: int, suite: str, lambda_value: float, variant: str) -> None:
    cfg = _common_mnist_cfg("edl_l1", seed=seed, suite=suite)
    cfg.experiment.method_variant = variant
    cfg.loss.name = "edl_fixed"
    cfg.loss.lambda_value = lambda_value
    cfg.loss.anneal_epochs = 200
    _train_and_eval(cfg)


def _run_sensitivity(seed: int, suite: str) -> None:
    for beta in (0.5, 1.0, 2.0):
        for gamma in (0.5, 1.0, 2.0):
            cfg = _common_mnist_cfg("info_edl", seed=seed, suite=suite)
            cfg.experiment.method_variant = f"mnist_bg_b{beta}_g{gamma}"
            cfg.loss.info_type = "fisher"
            cfg.loss.gate_type = "exp"
            cfg.loss.detach_weight = True
            cfg.loss.beta = beta
            cfg.loss.gamma = gamma
            _train_and_eval(cfg)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--suite", default="mnist_additional_20260320")
    parser.add_argument("--run-fixed", action="store_true")
    parser.add_argument("--run-controller", action="store_true")
    parser.add_argument("--run-sensitivity", action="store_true")
    args = parser.parse_args(argv)

    if not any([args.run_fixed, args.run_controller, args.run_sensitivity]):
        parser.error("Enable at least one of --run-fixed, --run-controller, --run-sensitivity")

    for seed in args.seeds:
        if args.run_fixed:
            _run_fixed(seed=seed, suite=args.suite, lambda_value=1.0, variant="mnist_fixed_l1_matched")
        if args.run_controller:
            _run_controller(seed=seed, suite=args.suite)
        if args.run_sensitivity:
            _run_sensitivity(seed=seed, suite=args.suite)


if __name__ == "__main__":
    main()
