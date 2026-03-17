from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_config_path(official_root: Path, method: str, ways: int) -> Path:
    method = method.lower()
    if method == "edl":
        stem = f"{ways}w-edl.json"
    elif method == "iedl":
        stem = f"{ways}w-iedl.json"
    elif method == "infoedl":
        stem = f"{ways}w-infoedl.json"
    else:
        raise ValueError(f"Unsupported method={method}")
    return official_root / "configs" / "1_mini" / stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--method", required=True, choices=["edl", "iedl", "infoedl"])
    parser.add_argument("--ways", required=True, type=int, choices=[5, 10])
    parser.add_argument("--tasks", type=int, default=10000)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--split", default="novel")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--suffix", default="official")
    parser.add_argument("--dump-period", type=int, default=10000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--optimizer-name", default="adam", choices=["adam", "lbfgs"])
    parser.add_argument("--info-beta", type=float, default=None)
    parser.add_argument("--info-gamma", type=float, default=None)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--lbfgs-line-search-fn", default="none", choices=["none", "strong_wolfe"])
    parser.add_argument("--adam-lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    config_path = _resolve_config_path(official_root, method=args.method, ways=args.ways)
    sys.path.insert(0, str(official_root))

    from main import main as official_main  # type: ignore

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["config_id"] = args.config_id
    cfg["suffix"] = args.suffix
    cfg["n_tasks"] = args.tasks
    cfg["task_start"] = args.task_start
    cfg["split_list"] = [args.split]
    cfg["use_wandb"] = args.use_wandb
    cfg["dump_period"] = args.dump_period
    cfg["torch_threads"] = args.torch_threads
    cfg["optimizer_name"] = args.optimizer_name
    if args.info_beta is not None:
        cfg["info_beta"] = args.info_beta
    if args.info_gamma is not None:
        cfg["info_gamma"] = args.info_gamma
    cfg["lbfgs_lr"] = args.lbfgs_lr
    cfg["lbfgs_line_search_fn"] = None if args.lbfgs_line_search_fn == "none" else args.lbfgs_line_search_fn
    cfg["adam_lr"] = args.adam_lr
    cfg["grad_clip_norm"] = args.grad_clip_norm
    cfg["results_dir"] = str(Path(args.results_dir).resolve())
    cfg["cache_dir"] = str((official_root / "cache").resolve())
    cfg["features_dir"] = str((official_root / "features").resolve())
    official_main(cfg)


if __name__ == "__main__":
    main()
