from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PATTERN = re.compile(r"runs/([^/]+)/seed_(\d+)/(\d{8}T\d+Z)/metrics\.jsonl$")

METHOD_DISPLAY = {
    "edl_l1": "EDL (lambda=1.0)",
    "edl_l01": "EDL (lambda=0.1)",
    "edl_l0001": "EDL (lambda=0.001)",
    "info_edl": "Info-EDL",
    "iedl_ref": "I-EDL Ref",
}


@dataclass
class ConfRow:
    method: str
    seed: int
    dataset: str
    score_type: str
    calibration_type: str
    ts: str
    accuracy: float
    aupr: float


def _mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    valid = [v for v in vals if v == v]
    if not valid:
        return float("nan"), float("nan")
    mu = sum(valid) / len(valid)
    sd = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    return mu, sd


def _format_percent(mean: float, std: float) -> str:
    return f"{mean * 100.0:.2f} +/- {std * 100.0:.2f}"


def _load_latest_conf_rows(runs_root: Path) -> List[ConfRow]:
    latest: Dict[Tuple[str, int, str, str, str], ConfRow] = {}
    for path in glob.glob(str(runs_root / "*" / "seed_*" / "*" / "metrics.jsonl")):
        match = PATTERN.search(path.replace("\\", "/"))
        if not match:
            continue
        method, seed, ts = match.group(1), int(match.group(2)), match.group(3)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("split") != "conf_eval":
                    continue
                metrics = payload.get("metrics", {})
                row = ConfRow(
                    method=method,
                    seed=seed,
                    dataset=str(payload.get("dataset", "unknown")),
                    score_type=str(payload.get("score_type", "maxp")),
                    calibration_type=str(payload.get("calibration_type", "none")),
                    ts=ts,
                    accuracy=float(metrics.get("accuracy", float("nan"))),
                    aupr=float(metrics.get("aupr", float("nan"))),
                )
                key = (row.method, row.seed, row.dataset, row.score_type, row.calibration_type)
                if key not in latest or row.ts > latest[key].ts:
                    latest[key] = row
    return sorted(latest.values(), key=lambda row: (row.method, row.seed, row.dataset, row.score_type))


def _build_project_rows(rows: List[ConfRow], method_order: List[str], dataset: str) -> List[dict]:
    grouped: Dict[str, Dict[str, List[ConfRow]]] = {}
    for row in rows:
        if row.dataset != dataset:
            continue
        grouped.setdefault(row.method, {}).setdefault(row.score_type, []).append(row)

    out: List[dict] = []
    for method in method_order:
        score_rows = grouped.get(method, {})
        maxp_rows = score_rows.get("maxp", [])
        maxalpha_rows = score_rows.get("maxalpha", [])
        acc_rows = maxp_rows or maxalpha_rows
        maxp_mean, maxp_std = _mean_std(item.aupr for item in maxp_rows)
        maxalpha_mean, maxalpha_std = _mean_std(item.aupr for item in maxalpha_rows)
        acc_mean, acc_std = _mean_std(item.accuracy for item in acc_rows)
        out.append(
            {
                "method_key": method,
                "display_name": METHOD_DISPLAY.get(method, method),
                "source": "project",
                "maxp_aupr": _format_percent(maxp_mean, maxp_std) if maxp_rows else "-",
                "maxalpha_aupr": _format_percent(maxalpha_mean, maxalpha_std) if maxalpha_rows else "-",
                "accuracy": _format_percent(acc_mean, acc_std) if acc_rows else "-",
            }
        )
    return out


def _load_reference_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_rows(reference_rows: List[dict], project_rows: List[dict]) -> List[dict]:
    seen = {row["method_key"] for row in reference_rows}
    out = list(reference_rows)
    for row in project_rows:
        if row["method_key"] not in seen:
            out.append(row)
    return out


def _write_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = ["method_key", "display_name", "source", "maxp_aupr", "maxalpha_aupr", "accuracy"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[dict], out_path: Path) -> None:
    lines = [
        "# Table 3 Style CIFAR10 Summary",
        "",
        "| Method | Max.P AUPR | Max.alpha AUPR | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['maxp_aupr']} | {row['maxalpha_aupr']} | {row['accuracy']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(rows: List[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CIFAR10 misclassification detection AUPR and classification accuracy.}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Max.P & Max.$\alpha$ & Acc. \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row["display_name"], row["maxp_aupr"], row["maxalpha_aupr"], row["accuracy"]]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--reference-csv", default="configs/paper/iedl_table3_reference_template.csv")
    parser.add_argument("--out-dir", default="results/paper_tables/iedl_table3")
    parser.add_argument(
        "--methods",
        default="edl_l1,edl_l01,edl_l0001,iedl_ref,info_edl",
        help="Comma-separated project methods to include.",
    )
    args = parser.parse_args()

    rows = _load_latest_conf_rows(Path(args.runs))
    method_order = [item.strip() for item in args.methods.split(",") if item.strip()]
    project_rows = _build_project_rows(rows, method_order=method_order, dataset=args.dataset)
    merged_rows = _merge_rows(_load_reference_rows(Path(args.reference_csv)), project_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(merged_rows, out_dir / "table3_iedl_style.csv")
    _write_markdown(merged_rows, out_dir / "table3_iedl_style.md")
    _write_latex(merged_rows, out_dir / "table3_iedl_style.tex")


if __name__ == "__main__":
    main()
