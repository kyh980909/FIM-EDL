from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PATTERN = re.compile(r"runs/([^/]+)/seed_(\d+)/(\d{8}T\d+Z)/metrics\.jsonl$")


@dataclass
class FewshotRow:
    method: str
    method_variant: str
    seed: int
    dataset: str
    score_type: str
    metric_family: str
    ts: str
    accuracy: float
    accuracy_ci95: float
    aupr: float
    aupr_ci95: float
    auroc: float
    auroc_ci95: float
    fpr95: float
    fpr95_ci95: float
    way: int
    shot: int
    episodes: int
    eval_split: str
    query_per_class_mean: float


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    valid = [v for v in vals if v == v]
    if not valid:
        return float("nan"), float("nan")
    mean = sum(valid) / len(valid)
    std = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    return mean, std


def _load_latest_rows(runs_root: Path) -> List[FewshotRow]:
    latest: Dict[Tuple[str, int, str, str, str, int, int, str], FewshotRow] = {}
    for path in glob.glob(str(runs_root / "*" / "seed_*" / "*" / "metrics.jsonl")):
        match = PATTERN.search(path.replace("\\", "/"))
        if not match:
            continue
        method, seed, ts = match.group(1), int(match.group(2)), match.group(3)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != "fewshot_eval":
                    continue
                metrics = row.get("metrics", {})
                extra = row.get("extra", {})
                parsed = FewshotRow(
                    method=method,
                    method_variant=str(row.get("method_variant", method)),
                    seed=seed,
                    dataset=str(row.get("dataset", "unknown")),
                    score_type=str(row.get("score_type", "")),
                    metric_family=str(extra.get("metric_family", "unknown")),
                    ts=ts,
                    accuracy=float(metrics.get("accuracy", float("nan"))),
                    accuracy_ci95=float(extra.get("accuracy_ci95", float("nan"))),
                    aupr=float(metrics.get("aupr", float("nan"))),
                    aupr_ci95=float(extra.get("aupr_ci95", float("nan"))),
                    auroc=float(metrics.get("auroc", float("nan"))),
                    auroc_ci95=float(extra.get("auroc_ci95", float("nan"))),
                    fpr95=float(metrics.get("fpr95", float("nan"))),
                    fpr95_ci95=float(extra.get("fpr95_ci95", float("nan"))),
                    way=int(extra.get("way", 0)),
                    shot=int(extra.get("shot", 0)),
                    episodes=int(extra.get("episodes", 0)),
                    eval_split=str(extra.get("eval_split", "")),
                    query_per_class_mean=float(extra.get("query_per_class_mean", float("nan"))),
                )
                key = (
                    parsed.method,
                    parsed.seed,
                    parsed.dataset,
                    parsed.score_type,
                    parsed.metric_family,
                    parsed.way,
                    parsed.shot,
                    parsed.eval_split,
                )
                if key not in latest or parsed.ts > latest[key].ts:
                    latest[key] = parsed
    return sorted(
        latest.values(),
        key=lambda item: (item.method, item.metric_family, item.dataset, item.score_type, item.seed),
    )


def _write_latest(rows: List[FewshotRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "method_variant",
                "seed",
                "metric_family",
                "dataset",
                "score_type",
                "timestamp",
                "way",
                "shot",
                "episodes",
                "eval_split",
                "query_per_class_mean",
                "accuracy",
                "accuracy_ci95",
                "aupr",
                "aupr_ci95",
                "auroc",
                "auroc_ci95",
                "fpr95",
                "fpr95_ci95",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.method,
                    row.method_variant,
                    row.seed,
                    row.metric_family,
                    row.dataset,
                    row.score_type,
                    row.ts,
                    row.way,
                    row.shot,
                    row.episodes,
                    row.eval_split,
                    f"{row.query_per_class_mean:.6f}",
                    f"{row.accuracy:.6f}",
                    f"{row.accuracy_ci95:.6f}",
                    f"{row.aupr:.6f}",
                    f"{row.aupr_ci95:.6f}",
                    f"{row.auroc:.6f}",
                    f"{row.auroc_ci95:.6f}",
                    f"{row.fpr95:.6f}",
                    f"{row.fpr95_ci95:.6f}",
                ]
            )


def _write_summary(rows: List[FewshotRow], out_path: Path) -> None:
    groups: Dict[Tuple[str, str, str, int, int, str], List[FewshotRow]] = {}
    for row in rows:
        groups.setdefault((row.method, row.metric_family, row.dataset, row.way, row.shot, row.score_type), []).append(
            row
        )

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "metric_family",
                "dataset",
                "score_type",
                "way",
                "shot",
                "accuracy_mean",
                "accuracy_std",
                "aupr_mean",
                "aupr_std",
                "auroc_mean",
                "auroc_std",
                "fpr95_mean",
                "fpr95_std",
                "n",
            ]
        )
        for key in sorted(groups):
            sub = groups[key]
            acc_m, acc_s = _mean_std([row.accuracy for row in sub])
            aupr_m, aupr_s = _mean_std([row.aupr for row in sub])
            auroc_m, auroc_s = _mean_std([row.auroc for row in sub])
            fpr95_m, fpr95_s = _mean_std([row.fpr95 for row in sub])
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    key[5],
                    key[3],
                    key[4],
                    f"{acc_m:.6f}",
                    f"{acc_s:.6f}",
                    f"{aupr_m:.6f}",
                    f"{aupr_s:.6f}",
                    f"{auroc_m:.6f}",
                    f"{auroc_s:.6f}",
                    f"{fpr95_m:.6f}",
                    f"{fpr95_s:.6f}",
                    len(sub),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/fewshot")
    args = parser.parse_args()

    runs_root = Path(args.runs)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _load_latest_rows(runs_root)
    if not rows:
        raise SystemExit(f"No few-shot rows found under: {runs_root}")

    latest_path = out_root / "latest_fewshot_rows.csv"
    summary_path = out_root / "summary_mean_std.csv"
    _write_latest(rows, latest_path)
    _write_summary(rows, summary_path)

    manifest = {
        "rows": len(rows),
        "outputs": [str(latest_path), str(summary_path)],
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved few-shot artifacts to: {out_root}")


if __name__ == "__main__":
    main()
