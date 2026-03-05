from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import OmegaConf

from src.reporting.schema import get_results_schema_version


class LocalCollector:
    def __init__(self, cfg) -> None:
        method = cfg.experiment.name
        seed = cfg.seed
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(cfg.logging.local_dir) / method / f"seed_{seed}" / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        resolved = OmegaConf.to_container(cfg, resolve=True)
        (self.run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        (self.run_dir / "env.json").write_text(json.dumps(self._env_info(), indent=2), encoding="utf-8")

        cfg_bytes = json.dumps(resolved, sort_keys=True).encode("utf-8")
        self.config_hash = hashlib.sha256(cfg_bytes).hexdigest()[:12]
        self.git_commit = self._git_commit()

    def _git_commit(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return "unknown"

    def _env_info(self) -> Dict[str, Any]:
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "results_schema_version": get_results_schema_version(),
        }

    def append_metric(
        self,
        method: str,
        seed: int,
        dataset: str,
        split: str,
        metrics: Dict[str, float],
        method_variant: str = "",
        score_type: str = "",
        calibration_type: str = "",
        step: int | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        row = {
            "results_schema_version": get_results_schema_version(),
            "method": method,
            "seed": seed,
            "dataset": dataset,
            "split": split,
            "metrics": metrics,
            "method_variant": method_variant,
            "score_type": score_type,
            "calibration_type": calibration_type,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "step": step,
            "extra": extra or {},
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        out = {
            "results_schema_version": get_results_schema_version(),
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "summary": summary,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
