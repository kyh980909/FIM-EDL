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
class EvalRow:
    method: str
    method_variant: str
    seed: int
    dataset: str
    score_type: str
    calibration_type: str
    ts: str
    accuracy: float
    nll: float
    ece: float
    aurc: float
    auroc: float
    aupr: float
    fpr95: float


def _load_latest_eval_rows(runs_root: Path) -> List[EvalRow]:
    latest: Dict[Tuple[str, int, str, str, str], EvalRow] = {}
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
                    method_variant=str(d.get("method_variant", method)),
                    seed=seed,
                    dataset=d.get("dataset", "unknown"),
                    score_type=str(d.get("score_type", "vacuity")),
                    calibration_type=str(d.get("calibration_type", "none")),
                    ts=ts,
                    accuracy=float(metrics.get("accuracy", float("nan"))),
                    nll=float(metrics.get("nll", float("nan"))),
                    ece=float(metrics.get("ece", float("nan"))),
                    aurc=float(metrics.get("aurc", float("nan"))),
                    auroc=float(metrics.get("auroc", float("nan"))),
                    aupr=float(metrics.get("aupr", float("nan"))),
                    fpr95=float(metrics.get("fpr95", float("nan"))),
                )
                key = (row.method, row.seed, row.dataset, row.score_type, row.calibration_type)
                if key not in latest or row.ts > latest[key].ts:
                    latest[key] = row
    return sorted(latest.values(), key=lambda r: (r.method, r.seed, r.dataset, r.score_type))


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    valid = [v for v in vals if v == v]
    if not valid:
        return float("nan"), float("nan")
    mu = sum(valid) / len(valid)
    sd = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    return mu, sd


def _write_latest_csv(rows: List[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "method_variant",
                "seed",
                "dataset",
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
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.method_variant,
                    r.seed,
                    r.dataset,
                    r.score_type,
                    r.calibration_type,
                    r.ts,
                    f"{r.accuracy:.6f}",
                    f"{r.nll:.6f}",
                    f"{r.ece:.6f}",
                    f"{r.aurc:.6f}",
                    f"{r.auroc:.6f}",
                    f"{r.aupr:.6f}",
                    f"{r.fpr95:.6f}",
                ]
            )


def _write_summary_csv(rows: List[EvalRow], out_path: Path) -> None:
    groups: Dict[Tuple[str, str, str, str], List[EvalRow]] = {}
    for r in rows:
        groups.setdefault((r.method, r.dataset, r.score_type, r.calibration_type), []).append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "dataset",
                "score_type",
                "calibration_type",
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
        for (method, dataset, score_type, cal_type) in sorted(groups.keys()):
            sub = groups[(method, dataset, score_type, cal_type)]
            acc_m, acc_s = _mean_std([r.accuracy for r in sub])
            nll_m, nll_s = _mean_std([r.nll for r in sub])
            ece_m, ece_s = _mean_std([r.ece for r in sub])
            aurc_m, aurc_s = _mean_std([r.aurc for r in sub])
            au_m, au_s = _mean_std([r.auroc for r in sub])
            ap_m, ap_s = _mean_std([r.aupr for r in sub])
            fp_m, fp_s = _mean_std([r.fpr95 for r in sub])
            w.writerow(
                [
                    method,
                    dataset,
                    score_type,
                    cal_type,
                    f"{acc_m:.6f}",
                    f"{acc_s:.6f}",
                    f"{nll_m:.6f}",
                    f"{nll_s:.6f}",
                    f"{ece_m:.6f}",
                    f"{ece_s:.6f}",
                    f"{aurc_m:.6f}",
                    f"{aurc_s:.6f}",
                    f"{au_m:.6f}",
                    f"{au_s:.6f}",
                    f"{ap_m:.6f}",
                    f"{ap_s:.6f}",
                    f"{fp_m:.6f}",
                    f"{fp_s:.6f}",
                    len(sub),
                ]
            )


def _write_summary_md(rows: List[EvalRow], out_path: Path) -> None:
    groups: Dict[Tuple[str, str, str, str], List[EvalRow]] = {}
    for r in rows:
        groups.setdefault((r.method, r.dataset, r.score_type, r.calibration_type), []).append(r)

    lines = [
        "# Eval Summary (Latest per method/seed/dataset/score)",
        "",
        "| Method | Dataset | Score | Calib | Accuracy | AUROC | AUPR | FPR95 | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for (method, dataset, score_type, cal_type) in sorted(groups.keys()):
        sub = groups[(method, dataset, score_type, cal_type)]
        acc_m, acc_s = _mean_std([r.accuracy for r in sub])
        au_m, au_s = _mean_std([r.auroc for r in sub])
        ap_m, ap_s = _mean_std([r.aupr for r in sub])
        fp_m, fp_s = _mean_std([r.fpr95 for r in sub])
        lines.append(
            f"| {method} | {dataset} | {score_type} | {cal_type} | {acc_m:.4f} ± {acc_s:.4f} | "
            f"{au_m:.4f} ± {au_s:.4f} | {ap_m:.4f} ± {ap_s:.4f} | {fp_m:.4f} ± {fp_s:.4f} | {len(sub)} |"
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
