"""Config smoke checks for preset/method additions."""

from pathlib import Path

import yaml


def test_edl_l0001_experiment_exists() -> None:
    path = Path("configs/experiment/edl_l0001.yaml")
    assert path.exists()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["loss"]["name"] == "edl_fixed"
    assert abs(float(cfg["loss"]["lambda_value"]) - 0.001) < 1e-12


def test_core_repro_contains_new_methods() -> None:
    cfg = yaml.safe_load(Path("configs/preset/core_repro.yaml").read_text(encoding="utf-8"))
    methods = list(cfg["methods"])
    assert "edl_l0001" in methods
    assert "iedl_ref" in methods
