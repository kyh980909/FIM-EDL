from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PATTERN = re.compile(r"runs/([^/]+)/seed_(\d+)/(\d{8}T\d+Z)/summary\.json$")

DEFAULT_SETTINGS = [(5, 1), (5, 5), (5, 20), (10, 1), (10, 5), (10, 20)]

METHOD_DISPLAY = {
    "edl_l1": "EDL (lambda=1.0)",
    "edl_l01": "EDL (lambda=0.1)",
    "edl_l0001": "EDL (lambda=0.001)",
    "info_edl": "Info-EDL",
    "iedl_ref": "I-EDL Ref",
    "edl_official": "EDL (official rerun)",
    "iedl_official": "I-EDL (official rerun)",
    "info_edl_official": "Info-EDL (official rerun)",
}


@dataclass
class FewshotSummary:
    method: str
    seed: int
    ts: str
    way: int
    shot: int
    episodes: int
    eval_split: str
    accuracy_mean: float
    accuracy_ci95: float
    conf_maxalpha_mean: float
    conf_maxalpha_ci95: float
    ood_alpha0_mean: float
    ood_alpha0_ci95: float


def _iter_summary_records(runs_root: Path) -> Iterable[FewshotSummary]:
    for path_str in glob.glob(str(runs_root / "*" / "seed_*" / "*" / "summary.json")):
        path = Path(path_str)
        match = PATTERN.search(path_str.replace("\\", "/"))
        if not match:
            continue
        method, seed, ts = match.group(1), int(match.group(2)), match.group(3)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        fewshot = summary.get("fewshot")
        if not isinstance(fewshot, dict):
            continue
        conf = summary.get("confidence_aupr", {})
        ood = summary.get("ood_aupr", {})
        acc = summary.get("accuracy", {})
        conf_maxalpha = conf.get("maxalpha", {})
        ood_alpha0 = ood.get("alpha0", {})
        yield FewshotSummary(
            method=method,
            seed=seed,
            ts=ts,
            way=int(fewshot.get("way", 0)),
            shot=int(fewshot.get("shot", 0)),
            episodes=int(fewshot.get("episodes", 0)),
            eval_split=str(fewshot.get("eval_split", "")),
            accuracy_mean=float(acc.get("mean", float("nan"))),
            accuracy_ci95=float(acc.get("ci95", float("nan"))),
            conf_maxalpha_mean=float(conf_maxalpha.get("mean", float("nan"))),
            conf_maxalpha_ci95=float(conf_maxalpha.get("ci95", float("nan"))),
            ood_alpha0_mean=float(ood_alpha0.get("mean", float("nan"))),
            ood_alpha0_ci95=float(ood_alpha0.get("ci95", float("nan"))),
        )


def _pick_best(records: Iterable[FewshotSummary]) -> Dict[Tuple[str, int, int, int, str], FewshotSummary]:
    best: Dict[Tuple[str, int, int, int, str], FewshotSummary] = {}
    for record in records:
        key = (record.method, record.seed, record.way, record.shot, record.eval_split)
        prev = best.get(key)
        if prev is None:
            best[key] = record
            continue
        if record.episodes > prev.episodes or (record.episodes == prev.episodes and record.ts > prev.ts):
            best[key] = record
    return best


def _format_percent(mean: float, ci95: float) -> str:
    if not math.isfinite(mean):
        return "-"
    if not math.isfinite(ci95):
        return f"{mean * 100.0:.2f}"
    return f"{mean * 100.0:.2f} +/- {ci95 * 100.0:.2f}"


def _settings_from_arg(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        way_str, shot_str = item.split("x", 1)
        out.append((int(way_str), int(shot_str)))
    return out


def _build_rows(
    best: Dict[Tuple[str, int, int, int, str], FewshotSummary],
    method_order: List[str],
    settings: List[Tuple[int, int]],
    eval_split: str,
) -> List[dict]:
    rows: List[dict] = []
    for method in method_order:
        row = {
            "method_key": method,
            "display_name": METHOD_DISPLAY.get(method, method),
            "source": "project",
        }
        for way, shot in settings:
            sub = [item for key, item in best.items() if key[0] == method and key[2] == way and key[3] == shot and key[4] == eval_split]
            if not sub:
                row[f"{way}way_{shot}shot_acc"] = "-"
                row[f"{way}way_{shot}shot_conf"] = "-"
                row[f"{way}way_{shot}shot_ood"] = "-"
                row[f"{way}way_{shot}shot_episodes"] = "-"
                continue
            chosen = sorted(sub, key=lambda item: (item.episodes, item.ts), reverse=True)[0]
            row[f"{way}way_{shot}shot_acc"] = _format_percent(chosen.accuracy_mean, chosen.accuracy_ci95)
            row[f"{way}way_{shot}shot_conf"] = _format_percent(chosen.conf_maxalpha_mean, chosen.conf_maxalpha_ci95)
            row[f"{way}way_{shot}shot_ood"] = _format_percent(chosen.ood_alpha0_mean, chosen.ood_alpha0_ci95)
            row[f"{way}way_{shot}shot_episodes"] = str(chosen.episodes)
        rows.append(row)
    return rows


def _fieldnames(settings: List[Tuple[int, int]]) -> List[str]:
    fieldnames = ["method_key", "display_name", "source"]
    for way, shot in settings:
        fieldnames.extend(
            [
                f"{way}way_{shot}shot_acc",
                f"{way}way_{shot}shot_conf",
                f"{way}way_{shot}shot_ood",
                f"{way}way_{shot}shot_episodes",
            ]
        )
    return fieldnames


def _load_reference_rows(path: Path, settings: List[Tuple[int, int]]) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    keep = set(_fieldnames(settings))
    rows: List[dict] = []
    for raw in raw_rows:
        row = {key: raw.get(key, "") for key in keep}
        row["method_key"] = raw["method_key"]
        row["display_name"] = raw["display_name"]
        row["source"] = raw.get("source", "reference")
        rows.append(row)
    return rows


def _load_official_summary_rows(
    path: Path,
    method_order: List[str],
    settings: List[Tuple[int, int]],
    eval_split: str,
) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    grouped: Dict[Tuple[str, int, int, str], List[dict]] = {}
    for raw in raw_rows:
        try:
            key = (
                raw["method_key"],
                int(raw["way"]),
                int(raw["shot"]),
                raw["eval_split"],
            )
        except (KeyError, ValueError):
            continue
        grouped.setdefault(key, []).append(raw)

    rows: List[dict] = []
    for method in method_order:
        candidates = [key for key in grouped if key[0] == method]
        if not candidates:
            continue
        row = {
            "method_key": method,
            "display_name": METHOD_DISPLAY.get(method, method),
            "source": "official_rerun",
        }
        for way, shot in settings:
            sub = grouped.get((method, way, shot, eval_split), [])
            if not sub:
                row[f"{way}way_{shot}shot_acc"] = "-"
                row[f"{way}way_{shot}shot_conf"] = "-"
                row[f"{way}way_{shot}shot_ood"] = "-"
                row[f"{way}way_{shot}shot_episodes"] = "-"
                continue
            chosen = sorted(
                sub,
                key=lambda item: (int(item.get("episodes", "0")), item.get("file_mtime", "0")),
                reverse=True,
            )[0]
            row[f"{way}way_{shot}shot_acc"] = _format_percent(
                float(chosen["accuracy_mean"]), float(chosen["accuracy_ci95"])
            )
            row[f"{way}way_{shot}shot_conf"] = _format_percent(
                float(chosen["conf_maxalpha_mean"]), float(chosen["conf_maxalpha_ci95"])
            )
            row[f"{way}way_{shot}shot_ood"] = _format_percent(
                float(chosen["ood_alpha0_mean"]), float(chosen["ood_alpha0_ci95"])
            )
            row[f"{way}way_{shot}shot_episodes"] = chosen["episodes"]
        rows.append(row)
    return rows


def _merge_rows(reference_rows: List[dict], project_rows: List[dict]) -> List[dict]:
    seen = {row["method_key"] for row in reference_rows}
    merged = list(reference_rows)
    for row in project_rows:
        if row["method_key"] not in seen:
            merged.append(row)
            seen.add(row["method_key"])
    return merged


def _write_csv(rows: List[dict], settings: List[Tuple[int, int]], out_path: Path) -> None:
    fieldnames = _fieldnames(settings)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[dict], settings: List[Tuple[int, int]], out_path: Path) -> None:
    header = ["Method"]
    for way, shot in settings:
        header.extend([f"{way}-Way {shot}-Shot Acc.", "Conf. (Max.alpha)", "OOD (alpha0)"])
    lines = [
        "# Table 4 Style Few-Shot Summary",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in range(len(header) - 1)]) + " |",
    ]
    for row in rows:
        values = [row["display_name"]]
        for way, shot in settings:
            values.extend(
                [
                    row[f"{way}way_{shot}shot_acc"],
                    row[f"{way}way_{shot}shot_conf"],
                    row[f"{way}way_{shot}shot_ood"],
                ]
            )
        lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(rows: List[dict], settings: List[Tuple[int, int]], out_path: Path) -> None:
    cols = "l" + "ccc" * len(settings)
    group_header = " & ".join(
        [rf"\multicolumn{{3}}{{c}}{{{way}-Way {shot}-Shot}}" for way, shot in settings]
    )
    metric_header = " & ".join(["Acc. & Conf. (Max.$\\alpha$) & OOD ($\\alpha_0$)"] * len(settings))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Few-shot classification accuracy and AUPR scores in Table 4 style.}",
        r"\small",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        "Method & " + group_header + r" \\",
        " & " + metric_header + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row["display_name"]]
        for way, shot in settings:
            values.extend(
                [
                    row[f"{way}way_{shot}shot_acc"],
                    row[f"{way}way_{shot}shot_conf"],
                    row[f"{way}way_{shot}shot_ood"],
                ]
            )
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out-dir", default="results/paper_tables/iedl_table4")
    parser.add_argument("--methods", default="info_edl", help="Comma-separated project methods to include.")
    parser.add_argument("--reference-csv", default="configs/paper/iedl_table4_reference_template.csv")
    parser.add_argument(
        "--official-summary-csv",
        default="",
        help="Optional summary_mean_std.csv produced by export_official_fewshot_results.py",
    )
    parser.add_argument(
        "--settings",
        default="5x1,5x5,5x20,10x1,10x5,10x20",
        help="Comma-separated way x shot settings, for example 5x1,5x5,10x1",
    )
    parser.add_argument("--eval-split", default="test")
    args = parser.parse_args()

    method_order = [item.strip() for item in args.methods.split(",") if item.strip()]
    settings = _settings_from_arg(args.settings) if args.settings else list(DEFAULT_SETTINGS)
    best = _pick_best(_iter_summary_records(Path(args.runs)))
    project_rows = _build_rows(best=best, method_order=method_order, settings=settings, eval_split=args.eval_split)
    official_rows = _load_official_summary_rows(
        path=Path(args.official_summary_csv),
        method_order=method_order,
        settings=settings,
        eval_split=args.eval_split,
    ) if args.official_summary_csv else []
    reference_rows = _load_reference_rows(Path(args.reference_csv), settings=settings)
    rows = _merge_rows(reference_rows, official_rows)
    rows = _merge_rows(rows, project_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, settings, out_dir / "table4_iedl_style.csv")
    _write_markdown(rows, settings, out_dir / "table4_iedl_style.md")
    _write_latex(rows, settings, out_dir / "table4_iedl_style.tex")


if __name__ == "__main__":
    main()
