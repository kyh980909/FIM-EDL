from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PATTERN = re.compile(r"runs/([^/]+)/seed_(\d+)/(\d{8}T\d+Z)/metrics\.jsonl$")


@dataclass
class MetricRow:
    method: str
    method_variant: str
    seed: int
    dataset: str
    split: str
    score_type: str
    calibration_type: str
    ts: str
    metrics: Dict[str, float]


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    valid = [v for v in values if v == v]
    if not valid:
        return float("nan"), float("nan")
    mean = sum(valid) / len(valid)
    std = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    return mean, std


def _latest_rows(runs_root: Path) -> List[MetricRow]:
    latest: Dict[Tuple[str, str, int, str, str, str, str], MetricRow] = {}
    for path in glob.glob(str(runs_root / "*" / "seed_*" / "*" / "metrics.jsonl")):
        match = PATTERN.search(path.replace("\\", "/"))
        if not match:
            continue
        method, seed, ts = match.group(1), int(match.group(2)), match.group(3)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                row = MetricRow(
                    method=method,
                    method_variant=str(payload.get("method_variant") or method),
                    seed=seed,
                    dataset=str(payload.get("dataset", "unknown")),
                    split=str(payload.get("split", "")),
                    score_type=str(payload.get("score_type", "")),
                    calibration_type=str(payload.get("calibration_type", "none")),
                    ts=ts,
                    metrics={k: float(v) for k, v in payload.get("metrics", {}).items()},
                )
                key = (
                    row.method,
                    row.method_variant,
                    row.seed,
                    row.dataset,
                    row.split,
                    row.score_type,
                    row.calibration_type,
                )
                if key not in latest or row.ts > latest[key].ts:
                    latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            row.method,
            row.method_variant,
            row.seed,
            row.dataset,
            row.split,
            row.score_type,
        ),
    )


def _write_latest_csv(rows: List[MetricRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "method_variant",
                "seed",
                "dataset",
                "split",
                "score_type",
                "calibration_type",
                "timestamp",
                "accuracy",
                "nll",
                "ece",
                "aurc",
                "auroc",
                "aupr",
                "fpr95",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.method,
                    row.method_variant,
                    row.seed,
                    row.dataset,
                    row.split,
                    row.score_type,
                    row.calibration_type,
                    row.ts,
                    f"{row.metrics.get('accuracy', float('nan')):.6f}",
                    f"{row.metrics.get('nll', float('nan')):.6f}",
                    f"{row.metrics.get('ece', float('nan')):.6f}",
                    f"{row.metrics.get('aurc', float('nan')):.6f}",
                    f"{row.metrics.get('auroc', float('nan')):.6f}",
                    f"{row.metrics.get('aupr', float('nan')):.6f}",
                    f"{row.metrics.get('fpr95', float('nan')):.6f}",
                ]
            )


def _write_summary_csv(rows: List[MetricRow], out_path: Path) -> None:
    grouped: Dict[Tuple[str, str, str, str, str], List[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method, row.method_variant, row.dataset, row.split, row.score_type)].append(row)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "method_variant",
                "dataset",
                "split",
                "score_type",
                "accuracy_mean",
                "accuracy_std",
                "nll_mean",
                "nll_std",
                "ece_mean",
                "ece_std",
                "aurc_mean",
                "aurc_std",
                "auroc_mean",
                "auroc_std",
                "aupr_mean",
                "aupr_std",
                "fpr95_mean",
                "fpr95_std",
                "n",
            ]
        )
        for key in sorted(grouped):
            method, method_variant, dataset, split, score_type = key
            sub = grouped[key]
            acc_m, acc_s = _mean_std(row.metrics.get("accuracy", float("nan")) for row in sub)
            nll_m, nll_s = _mean_std(row.metrics.get("nll", float("nan")) for row in sub)
            ece_m, ece_s = _mean_std(row.metrics.get("ece", float("nan")) for row in sub)
            aurc_m, aurc_s = _mean_std(row.metrics.get("aurc", float("nan")) for row in sub)
            auroc_m, auroc_s = _mean_std(row.metrics.get("auroc", float("nan")) for row in sub)
            aupr_m, aupr_s = _mean_std(row.metrics.get("aupr", float("nan")) for row in sub)
            fpr95_m, fpr95_s = _mean_std(row.metrics.get("fpr95", float("nan")) for row in sub)
            writer.writerow(
                [
                    method,
                    method_variant,
                    dataset,
                    split,
                    score_type,
                    f"{acc_m:.6f}",
                    f"{acc_s:.6f}",
                    f"{nll_m:.6f}",
                    f"{nll_s:.6f}",
                    f"{ece_m:.6f}",
                    f"{ece_s:.6f}",
                    f"{aurc_m:.6f}",
                    f"{aurc_s:.6f}",
                    f"{auroc_m:.6f}",
                    f"{auroc_s:.6f}",
                    f"{aupr_m:.6f}",
                    f"{aupr_s:.6f}",
                    f"{fpr95_m:.6f}",
                    f"{fpr95_s:.6f}",
                    len(sub),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/additional_experiments")
    args = parser.parse_args()

    rows = _latest_rows(Path(args.runs))
    if not rows:
        raise SystemExit(f"No metrics found under {args.runs}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_latest_csv(rows, out_dir / "latest_rows.csv")
    _write_summary_csv(rows, out_dir / "summary_mean_std.csv")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "outputs": [
                    str(out_dir / "latest_rows.csv"),
                    str(out_dir / "summary_mean_std.csv"),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved additional experiment artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
