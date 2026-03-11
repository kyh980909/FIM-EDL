from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


OOD_TO_ID = {
    "kmnist": "mnist",
    "fmnist": "mnist",
    "fashionmnist": "mnist",
    "svhn": "cifar10",
    "cifar100": "cifar10",
}

PAIR_SPECS = [
    ("mnist", "kmnist", "MNIST -> KMNIST"),
    ("mnist", "fmnist", "MNIST -> FMNIST"),
    ("cifar10", "svhn", "CIFAR10 -> SVHN"),
    ("cifar10", "cifar100", "CIFAR10 -> CIFAR100"),
]

SCORE_COLUMNS = [("maxp", "Max.P"), ("alpha0", r"$\alpha_0$")]

METHOD_DISPLAY = {
    "edl_l1": "EDL (lambda=1.0)",
    "edl_l01": "EDL (lambda=0.1)",
    "edl_l0001": "EDL (lambda=0.001)",
    "info_edl": "Info-EDL",
    "iedl_ref": "I-EDL Ref",
}


@dataclass
class TableRow:
    method_key: str
    display_name: str
    cells: Dict[str, str]
    source: str


def _pair_key(id_dataset: str, ood_dataset: str, score_type: str) -> str:
    return f"{id_dataset}_to_{ood_dataset}_{score_type}"


def _format_percent(mean_str: str, std_str: str) -> str:
    mean = float(mean_str) * 100.0
    std = float(std_str) * 100.0
    return f"{mean:.2f} +/- {std:.2f}"


def _load_summary_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_project_rows(
    summary_rows: Iterable[dict],
    method_order: List[str],
) -> List[TableRow]:
    cells_by_method: Dict[str, Dict[str, str]] = {method: {} for method in method_order}
    for row in summary_rows:
        method = row["method"]
        if method not in cells_by_method:
            continue
        score_type = row["score_type"]
        if score_type not in {"maxp", "alpha0"}:
            continue
        ood_dataset = row["dataset"]
        id_dataset = OOD_TO_ID.get(ood_dataset)
        if id_dataset is None:
            continue
        key = _pair_key(id_dataset, ood_dataset, score_type)
        cells_by_method[method][key] = _format_percent(row["aupr_mean"], row["aupr_std"])

    rows: List[TableRow] = []
    for method in method_order:
        rows.append(
            TableRow(
                method_key=method,
                display_name=METHOD_DISPLAY.get(method, method),
                cells=cells_by_method[method],
                source="project",
            )
        )
    return rows


def _load_reference_rows(path: Path) -> List[TableRow]:
    if not path.exists():
        return []
    rows: List[TableRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            method_key = raw["method_key"]
            display_name = raw["display_name"]
            cells = {
                key: value
                for key, value in raw.items()
                if key not in {"method_key", "display_name", "source"} and value.strip()
            }
            rows.append(
                TableRow(
                    method_key=method_key,
                    display_name=display_name,
                    cells=cells,
                    source=raw.get("source", "reference"),
                )
            )
    return rows


def _all_column_keys() -> List[str]:
    keys: List[str] = []
    for id_dataset, ood_dataset, _ in PAIR_SPECS:
        for score_type, _ in SCORE_COLUMNS:
            keys.append(_pair_key(id_dataset, ood_dataset, score_type))
    return keys


def _merge_rows(reference_rows: List[TableRow], project_rows: List[TableRow]) -> List[TableRow]:
    seen = {row.method_key for row in reference_rows}
    merged = list(reference_rows)
    for row in project_rows:
        if row.method_key not in seen:
            merged.append(row)
    return merged


def _write_csv(rows: List[TableRow], out_path: Path) -> None:
    fieldnames = ["method_key", "display_name", "source"] + _all_column_keys()
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "method_key": row.method_key,
                "display_name": row.display_name,
                "source": row.source,
            }
            for key in _all_column_keys():
                payload[key] = row.cells.get(key, "-")
            writer.writerow(payload)


def _write_markdown(rows: List[TableRow], out_path: Path) -> None:
    lines = [
        "# Table 2 Style AUPR Summary",
        "",
        "| Method | MNIST -> KMNIST |  | MNIST -> FMNIST |  | CIFAR10 -> SVHN |  | CIFAR10 -> CIFAR100 |  |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "|  | Max.P | alpha0 | Max.P | alpha0 | Max.P | alpha0 | Max.P | alpha0 |",
    ]
    for row in rows:
        values = []
        for id_dataset, ood_dataset, _ in PAIR_SPECS:
            for score_type, label in SCORE_COLUMNS:
                del label
                values.append(row.cells.get(_pair_key(id_dataset, ood_dataset, score_type), "-"))
        lines.append("| " + " | ".join([row.display_name] + values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def _write_latex(rows: List[TableRow], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AUPR scores of OOD detection (mean $\pm$ standard deviation).}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{MNIST $\rightarrow$ KMNIST} & \multicolumn{2}{c}{MNIST $\rightarrow$ FMNIST} & \multicolumn{2}{c}{CIFAR10 $\rightarrow$ SVHN} & \multicolumn{2}{c}{CIFAR10 $\rightarrow$ CIFAR100} \\",
        r"Method & Max.P & $\alpha_0$ & Max.P & $\alpha_0$ & Max.P & $\alpha_0$ & Max.P & $\alpha_0$ \\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for id_dataset, ood_dataset, _ in PAIR_SPECS:
            for score_type, _ in SCORE_COLUMNS:
                values.append(_escape_latex(row.cells.get(_pair_key(id_dataset, ood_dataset, score_type), "-")))
        lines.append(" & ".join([_escape_latex(row.display_name)] + values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/eval/summary_mean_std.csv")
    parser.add_argument("--reference-csv", default="configs/paper/iedl_table2_reference_template.csv")
    parser.add_argument("--out-dir", default="results/paper_tables/iedl_table2")
    parser.add_argument(
        "--methods",
        default="iedl_ref,info_edl",
        help="Comma-separated project methods to include.",
    )
    args = parser.parse_args()

    summary_rows = _load_summary_rows(Path(args.summary_csv))
    method_order = [item.strip() for item in args.methods.split(",") if item.strip()]
    project_rows = _build_project_rows(summary_rows, method_order)
    reference_rows = _load_reference_rows(Path(args.reference_csv))
    rows = _merge_rows(reference_rows, project_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "table2_iedl_style.csv")
    _write_markdown(rows, out_dir / "table2_iedl_style.md")
    _write_latex(rows, out_dir / "table2_iedl_style.tex")


if __name__ == "__main__":
    main()
