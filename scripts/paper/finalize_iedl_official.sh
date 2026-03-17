#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yongho/FIM-EDL"
UV="uv run python"
LOG="$ROOT/outputs/logs/table23_cifar_iedl_official_finalize_20260310.log"
SUMMARY_TXT="$ROOT/results/paper_tables/iedl_table3/iedl_official_summary_20260310.txt"

cd "$ROOT"

while pgrep -f "run.py preset paper_iedl_cifar_ref_iedl_official|src.train experiment=iedl_ref|src.eval experiment=iedl_ref" >/dev/null; do
  sleep 60
done

$UV scripts/paper/export_eval_results.py --runs runs --out results/eval >>"$LOG" 2>&1
$UV scripts/paper/build_iedl_table2.py \
  --summary-csv results/eval/summary_mean_std.csv \
  --reference-csv configs/paper/iedl_table2_reference_template.csv \
  --out-dir results/paper_tables/iedl_table2 >>"$LOG" 2>&1
$UV scripts/paper/build_iedl_table3.py \
  --runs runs \
  --dataset cifar10 \
  --reference-csv configs/paper/iedl_table3_reference_template.csv \
  --out-dir results/paper_tables/iedl_table3 >>"$LOG" 2>&1

uv run python - <<'PY' >"$SUMMARY_TXT"
import csv
from pathlib import Path

root = Path("/home/yongho/FIM-EDL")
summary_path = root / "results/eval/summary_mean_std.csv"
table2_path = root / "results/paper_tables/iedl_table2/table2_iedl_style.md"
table3_path = root / "results/paper_tables/iedl_table3/table3_iedl_style.md"

rows = list(csv.DictReader(summary_path.open()))

def pick(method: str, dataset: str, score: str):
    for row in rows:
        if row["method"] == method and row["dataset"] == dataset and row["score_type"] == score:
            return row
    return None

iedl_svhn_alpha0 = pick("iedl_ref", "svhn", "alpha0")
iedl_svhn_maxp = pick("iedl_ref", "svhn", "maxp")
iedl_c100_alpha0 = pick("iedl_ref", "cifar100", "alpha0")
iedl_c100_maxp = pick("iedl_ref", "cifar100", "maxp")
info_svhn_alpha0 = pick("info_edl", "svhn", "alpha0")
info_c100_alpha0 = pick("info_edl", "cifar100", "alpha0")

def pct(row, key):
    return f"{float(row[key]) * 100:.2f}" if row else "n/a"

lines = [
    "Subject: CIFAR-10 I-EDL rerun summary",
    "",
    "CIFAR-10 I-EDL official-style rerun has completed.",
    "",
    "Latest I-EDL summary:",
    f"- SVHN OOD AUPR (Max.P): {pct(iedl_svhn_maxp, 'aupr_mean')} +/- {pct(iedl_svhn_maxp, 'aupr_std')}",
    f"- SVHN OOD AUPR (alpha0): {pct(iedl_svhn_alpha0, 'aupr_mean')} +/- {pct(iedl_svhn_alpha0, 'aupr_std')}",
    f"- CIFAR100 OOD AUPR (Max.P): {pct(iedl_c100_maxp, 'aupr_mean')} +/- {pct(iedl_c100_maxp, 'aupr_std')}",
    f"- CIFAR100 OOD AUPR (alpha0): {pct(iedl_c100_alpha0, 'aupr_mean')} +/- {pct(iedl_c100_alpha0, 'aupr_std')}",
    "",
    "Reference comparison inside current project:",
    f"- Info-EDL SVHN alpha0 AUPR: {pct(info_svhn_alpha0, 'aupr_mean')} +/- {pct(info_svhn_alpha0, 'aupr_std')}",
    f"- Info-EDL CIFAR100 alpha0 AUPR: {pct(info_c100_alpha0, 'aupr_mean')} +/- {pct(info_c100_alpha0, 'aupr_std')}",
    "",
    f"Updated tables:",
    f"- {table2_path}",
    f"- {table3_path}",
    f"- {summary_path}",
]
print("\n".join(lines))
PY

if command -v xdg-email >/dev/null 2>&1; then
  BODY_FILE="$SUMMARY_TXT"
  xdg-email --utf8 --subject "CIFAR-10 I-EDL rerun summary" --body "$(cat "$BODY_FILE")" yhkim98@mju.ac.kr >>"$LOG" 2>&1 || true
fi

echo "Finalize complete: $SUMMARY_TXT" >>"$LOG"
