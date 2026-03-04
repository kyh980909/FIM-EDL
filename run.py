from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def load_preset(name: str) -> Dict:
    path = Path("configs/preset") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preset experiment runner")
    parser.add_argument("mode", choices=["preset"], help="Run mode")
    parser.add_argument("preset", help="Preset name under configs/preset")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()

    preset = load_preset(args.preset)
    methods = preset["methods"]
    seeds = preset["seeds"]

    for method in methods:
        for seed in seeds:
            train_cmd = ["python", "-m", "src.train", f"experiment={method}", f"seed={seed}"] + args.overrides
            train_cmd[0] = sys.executable
            run_cmd(train_cmd)

    if preset.get("run_eval", True):
        for method in methods:
            for seed in seeds:
                run_root = Path("runs") / method / f"seed_{seed}"
                if not run_root.exists():
                    continue
                latest = sorted(run_root.glob("*/summary.json"))[-1]
                summary = yaml.safe_load(latest.read_text(encoding="utf-8"))
                ckpt = summary["summary"]["best_model_path"]
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "src.eval",
                    f"experiment={method}",
                    f"seed={seed}",
                    f"checkpoint={ckpt}",
                ] + args.overrides
                run_cmd(eval_cmd)

    if preset.get("build_artifacts", True):
        run_cmd(
            [
                sys.executable,
                "scripts/paper/build_paper_artifacts.py",
                "--input",
                "runs",
                "--out",
                "artifacts/paper",
            ]
        )


if __name__ == "__main__":
    main()
