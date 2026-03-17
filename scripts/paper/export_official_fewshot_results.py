from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LOSS_TO_METHOD = {
    "EDL": ("edl_official", "EDL (official rerun)"),
    "IEDL": ("iedl_official", "I-EDL (official rerun)"),
    "INFO_EDL": ("info_edl_official", "Info-EDL (official rerun)"),
}


@dataclass
class Aggregate:
    mean: float
    std: float
    ci95: float


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(values: List[float]) -> Aggregate:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return Aggregate(mean=float("nan"), std=float("nan"), ci95=float("nan"))
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        std = 0.0
    else:
        var = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
        std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(len(clean))
    return Aggregate(mean=mean, std=std, ci95=ci95)


def _iter_csv_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*.csv")):
        if path.name == "summary_mean_std.csv":
            continue
        yield path


def _summarize_file(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    grouped: Dict[Tuple[str, int, int, str], List[dict]] = defaultdict(list)
    for row in rows:
        try:
            key = (
                row["loss_type"].strip().upper(),
                int(row["n_ways"]),
                int(row["n_shots"]),
                row["split"].strip(),
            )
        except (KeyError, ValueError):
            continue
        grouped[key].append(row)

    out_rows: List[dict] = []
    for (loss_type, way, shot, split), items in sorted(grouped.items()):
        method_meta = LOSS_TO_METHOD.get(loss_type)
        if method_meta is None:
            continue
        method_key, display_name = method_meta
        acc = _aggregate([_safe_float(item.get("id_accuracy")) for item in items])
        conf = _aggregate([_safe_float(item.get("id_max_alp_apr")) for item in items])
        ood = _aggregate([_safe_float(item.get("ood_alpha0_apr")) for item in items])
        out_rows.append(
            {
                "method_key": method_key,
                "display_name": display_name,
                "protocol": "official_fsl_features",
                "source": "official_rerun",
                "source_csv": str(path),
                "file_mtime": str(path.stat().st_mtime_ns),
                "eval_split": split,
                "way": str(way),
                "shot": str(shot),
                "episodes": str(len(items)),
                "accuracy_mean": repr(acc.mean),
                "accuracy_std": repr(acc.std),
                "accuracy_ci95": repr(acc.ci95),
                "conf_maxalpha_mean": repr(conf.mean),
                "conf_maxalpha_std": repr(conf.std),
                "conf_maxalpha_ci95": repr(conf.ci95),
                "ood_alpha0_mean": repr(ood.mean),
                "ood_alpha0_std": repr(ood.std),
                "ood_alpha0_ci95": repr(ood.ci95),
            }
        )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Official few-shot CSV directory or a single CSV file.")
    parser.add_argument("--out", default="results/fewshot_official")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    scanned_paths: List[str] = []
    for path in _iter_csv_paths(results_root):
        scanned_paths.append(str(path))
        rows.extend(_summarize_file(path))

    rows.sort(
        key=lambda row: (
            row["method_key"],
            row["eval_split"],
            int(row["way"]),
            int(row["shot"]),
            int(row["episodes"]),
            row["file_mtime"],
        )
    )

    summary_path = out_dir / "summary_mean_std.csv"
    fieldnames = [
        "method_key",
        "display_name",
        "protocol",
        "source",
        "source_csv",
        "file_mtime",
        "eval_split",
        "way",
        "shot",
        "episodes",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_ci95",
        "conf_maxalpha_mean",
        "conf_maxalpha_std",
        "conf_maxalpha_ci95",
        "ood_alpha0_mean",
        "ood_alpha0_std",
        "ood_alpha0_ci95",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "results_dir": str(results_root),
                "files_scanned": len(scanned_paths),
                "summary_rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
