from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EVAL_PATTERN = re.compile(r"runs/([^/]+)/seed_(\d+)/(\d{8}T\d+Z)/metrics\.jsonl$")

METHOD_DISPLAY = {
    "info_edl": "Info-EDL",
    "iedl_ref": "I-EDL Ref",
}

OOD_TO_ID = {
    "kmnist": "mnist",
    "fmnist": "mnist",
    "fashionmnist": "mnist",
    "svhn": "cifar10",
    "cifar100": "cifar10",
}

TABLE2_DATASETS = ["mnist", "cifar10"]


def _mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    valid = [v for v in vals if v == v]
    if not valid:
        return float("nan"), float("nan")
    mu = sum(valid) / len(valid)
    sd = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    return mu, sd


def _format_percent(mean: float, std: float) -> str:
    return f"{mean * 100.0:.2f} +/- {std * 100.0:.2f}"


def _load_table2_summary(path: Path, methods: List[str]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in methods:
                continue
            id_dataset = OOD_TO_ID.get(row["dataset"])
            if id_dataset not in TABLE2_DATASETS:
                continue
            key = (method, id_dataset)
            if key in out:
                continue
            out[key] = _format_percent(float(row["ece_mean"]), float(row["ece_std"]))
    return out


def _load_conf_ece(runs_root: Path, methods: List[str], dataset: str) -> Dict[str, str]:
    latest: Dict[Tuple[str, int, str, str], Tuple[str, float]] = {}
    for path in glob.glob(str(runs_root / "*" / "seed_*" / "*" / "metrics.jsonl")):
        match = EVAL_PATTERN.search(path.replace("\\", "/"))
        if not match:
            continue
        method, seed, ts = match.group(1), int(match.group(2)), match.group(3)
        if method not in methods:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("split") != "conf_eval":
                    continue
                if str(payload.get("dataset")) != dataset:
                    continue
                calibration = str(payload.get("calibration_type", "none"))
                key = (method, seed, dataset, calibration)
                metrics = payload.get("metrics", {})
                ece = float(metrics.get("ece", float("nan")))
                if key not in latest or ts > latest[key][0]:
                    latest[key] = (ts, ece)

    grouped: Dict[str, List[float]] = {}
    for (method, _seed, _dataset, _calibration), (_ts, ece) in latest.items():
        grouped.setdefault(method, []).append(ece)

    return {
        method: _format_percent(*_mean_std(vals))
        for method, vals in grouped.items()
    }


def _write_table2_markdown(table2_rows: List[List[str]], out_path: Path) -> None:
    lines = [
        "# ECE Summary for Table 2 Models",
        "",
        "| Method | MNIST ECE | CIFAR10 ECE |",
        "|---|---:|---:|",
    ]
    for row in table2_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table2_csv(table2_rows: List[List[str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["display_name", "mnist_ece", "cifar10_ece"])
        writer.writerows(table2_rows)


def _write_table2_latex(table2_rows: List[List[str]], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Expected calibration error (ECE) for the models used in Table 2.}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & MNIST ECE & CIFAR10 ECE \\",
        r"\midrule",
    ]
    for row in table2_rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table3_markdown(table3_rows: List[List[str]], out_path: Path) -> None:
    lines = [
        "# ECE Summary for Table 3 Models",
        "",
        "| Method | CIFAR10 Conf-ECE |",
        "|---|---:|",
    ]
    for row in table3_rows:
        lines.append(f"| {row[0]} | {row[1]} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table3_csv(table3_rows: List[List[str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["display_name", "cifar10_conf_ece"])
        writer.writerows(table3_rows)


def _write_table3_latex(table3_rows: List[List[str]], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Expected calibration error (ECE) for the models used in Table 3.}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Method & CIFAR10 Conf-ECE \\",
        r"\midrule",
    ]
    for row in table3_rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/eval/summary_mean_std.csv")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out-dir", default="results/paper_tables/iedl_ece")
    parser.add_argument(
        "--methods",
        default="iedl_ref,info_edl",
        help="Comma-separated project methods to include.",
    )
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    table2_map = _load_table2_summary(Path(args.summary_csv), methods)
    table3_map = _load_conf_ece(Path(args.runs), methods, dataset="cifar10")

    table2_rows = [
        [
            METHOD_DISPLAY.get(method, method),
            table2_map.get((method, "mnist"), "-"),
            table2_map.get((method, "cifar10"), "-"),
        ]
        for method in methods
    ]
    table3_rows = [
        [METHOD_DISPLAY.get(method, method), table3_map.get(method, "-")]
        for method in methods
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_table2_csv(table2_rows, out_dir / "table2_ece.csv")
    _write_table2_markdown(table2_rows, out_dir / "table2_ece.md")
    _write_table2_latex(table2_rows, out_dir / "table2_ece.tex")
    _write_table3_csv(table3_rows, out_dir / "table3_ece.csv")
    _write_table3_markdown(table3_rows, out_dir / "table3_ece.md")
    _write_table3_latex(table3_rows, out_dir / "table3_ece.tex")


if __name__ == "__main__":
    main()
