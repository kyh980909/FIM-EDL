from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PATTERN = re.compile(r"runs/([^/]+)/seed_(\d+)/(\d{8}T\d+Z)/metrics\.jsonl$")


@dataclass
class EvalRow:
    method: str
    seed: int
    dataset: str
    ts: str
    accuracy: float
    auroc: float
    fpr95: float


def _load_latest_eval_rows(runs_root: Path) -> List[EvalRow]:
    latest: Dict[Tuple[str, int, str], EvalRow] = {}
    for path in glob.glob(str(runs_root / "*" / "seed_*" / "*" / "metrics.jsonl")):
        m = PATTERN.search(path.replace("\\", "/"))
        if not m:
            continue
        method, seed, ts = m.group(1), int(m.group(2)), m.group(3)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("split") != "eval":
                    continue
                metrics = d.get("metrics", {})
                row = EvalRow(
                    method=method,
                    seed=seed,
                    dataset=d.get("dataset", "unknown"),
                    ts=ts,
                    accuracy=float(metrics.get("accuracy", float("nan"))),
                    auroc=float(metrics.get("auroc", float("nan"))),
                    fpr95=float(metrics.get("fpr95", float("nan"))),
                )
                key = (row.method, row.seed, row.dataset)
                if key not in latest or row.ts > latest[key].ts:
                    latest[key] = row
    return sorted(latest.values(), key=lambda r: (r.method, r.seed, r.dataset))


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    mu = sum(vals) / len(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mu, sd


def _write_latest_csv(rows: List[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "dataset", "timestamp", "accuracy", "auroc", "fpr95"])
        for r in rows:
            w.writerow([r.method, r.seed, r.dataset, r.ts, f"{r.accuracy:.6f}", f"{r.auroc:.6f}", f"{r.fpr95:.6f}"])


def _write_summary_csv(rows: List[EvalRow], out_path: Path) -> None:
    groups: Dict[Tuple[str, str], List[EvalRow]] = {}
    for r in rows:
        groups.setdefault((r.method, r.dataset), []).append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "accuracy_mean", "accuracy_std", "auroc_mean", "auroc_std", "fpr95_mean", "fpr95_std", "n"])
        for (method, dataset) in sorted(groups.keys()):
            sub = groups[(method, dataset)]
            acc_m, acc_s = _mean_std([r.accuracy for r in sub])
            au_m, au_s = _mean_std([r.auroc for r in sub])
            fp_m, fp_s = _mean_std([r.fpr95 for r in sub])
            w.writerow([
                method,
                dataset,
                f"{acc_m:.6f}",
                f"{acc_s:.6f}",
                f"{au_m:.6f}",
                f"{au_s:.6f}",
                f"{fp_m:.6f}",
                f"{fp_s:.6f}",
                len(sub),
            ])


def _write_summary_md(rows: List[EvalRow], out_path: Path) -> None:
    groups: Dict[Tuple[str, str], List[EvalRow]] = {}
    for r in rows:
        groups.setdefault((r.method, r.dataset), []).append(r)

    lines = [
        "# Eval Summary (Latest per method/seed/dataset)",
        "",
        "| Method | Dataset | Accuracy (mean±std) | AUROC (mean±std) | FPR95 (mean±std) | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for (method, dataset) in sorted(groups.keys()):
        sub = groups[(method, dataset)]
        acc_m, acc_s = _mean_std([r.accuracy for r in sub])
        au_m, au_s = _mean_std([r.auroc for r in sub])
        fp_m, fp_s = _mean_std([r.fpr95 for r in sub])
        lines.append(
            f"| {method} | {dataset} | {acc_m:.4f} ± {acc_s:.4f} | {au_m:.4f} ± {au_s:.4f} | {fp_m:.4f} ± {fp_s:.4f} | {len(sub)} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/eval")
    args = parser.parse_args()

    runs_root = Path(args.runs)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _load_latest_eval_rows(runs_root)
    if not rows:
        raise SystemExit(f"No eval rows found under: {runs_root}")

    _write_latest_csv(rows, out_root / "latest_eval_rows.csv")
    _write_summary_csv(rows, out_root / "summary_mean_std.csv")
    _write_summary_md(rows, out_root / "summary.md")

    meta = {
        "rows": len(rows),
        "outputs": [
            str(out_root / "latest_eval_rows.csv"),
            str(out_root / "summary_mean_std.csv"),
            str(out_root / "summary.md"),
        ],
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved eval artifacts to: {out_root}")


if __name__ == "__main__":
    main()
