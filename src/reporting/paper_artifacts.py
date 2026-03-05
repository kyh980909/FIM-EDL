from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class PaperArtifactsBuilder:
    runs_dir: Path
    out_dir: Path

    def collect(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for path in self.runs_dir.rglob("metrics.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                rows.append(json.loads(line))
        if not rows:
            raise RuntimeError(f"No metrics found under {self.runs_dir}")
        return pd.DataFrame(rows)

    def build_main_table(self, df: pd.DataFrame) -> pd.DataFrame:
        exp = df[df["split"] == "eval"]
        flat = pd.json_normalize(exp.to_dict(orient="records"))
        for col in ["score_type", "calibration_type", "method_variant"]:
            if col not in flat.columns:
                flat[col] = "unknown"
        grp = flat.groupby(["method", "method_variant", "dataset", "score_type", "calibration_type"])
        metric_cols = [
            c
            for c in [
                "metrics.accuracy",
                "metrics.nll",
                "metrics.ece",
                "metrics.aurc",
                "metrics.auroc",
                "metrics.aupr",
                "metrics.fpr95",
            ]
            if c in flat.columns
        ]
        return grp[metric_cols].agg(["mean", "std"])

    def save_table(self, table: pd.DataFrame) -> None:
        table_dir = self.out_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        csv_path = table_dir / "table_main.csv"
        tex_path = table_dir / "table_main.tex"
        table.to_csv(csv_path)
        tex_path.write_text(table.to_latex(float_format=lambda x: f"{x:.4f}"), encoding="utf-8")

    def save_figures(self, df: pd.DataFrame) -> None:
        fig_dir = self.out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        flat = pd.json_normalize(df[df["split"] == "eval"].to_dict(orient="records"))

        if "score_type" not in flat.columns:
            flat["score_type"] = "vacuity"

        # 1) OOD AUROC (main comparison)
        if "metrics.auroc" in flat.columns:
            plt.figure(figsize=(7, 4))
            plot_df = (
                flat[flat["score_type"] == "vacuity"]
                .groupby(["method", "dataset"], as_index=False)["metrics.auroc"]
                .mean()
            )
            for method, sub in plot_df.groupby("method"):
                plt.plot(sub["dataset"], sub["metrics.auroc"], marker="o", label=method)
            plt.title("OOD AUROC by Method (score=vacuity)")
            plt.ylabel("AUROC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "main_ood_auroc.png")
            plt.savefig(fig_dir / "main_ood_auroc.pdf")
            plt.close()

        # 2) Risk-Coverage proxy from AURC (lower is better)
        if "metrics.aurc" in flat.columns:
            plt.figure(figsize=(7, 4))
            plot_df = (
                flat[flat["score_type"] == "maxp"]
                .groupby(["method"], as_index=False)["metrics.aurc"]
                .mean()
                .sort_values("metrics.aurc")
            )
            plt.bar(plot_df["method"], plot_df["metrics.aurc"])
            plt.title("Risk-Coverage Proxy (AURC, lower better)")
            plt.ylabel("AURC")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "risk_coverage_proxy.png")
            plt.savefig(fig_dir / "risk_coverage_proxy.pdf")
            plt.close()

        # 3) Seed dispersion on AUROC
        if "metrics.auroc" in flat.columns:
            plt.figure(figsize=(7, 4))
            box_df = flat[flat["score_type"] == "vacuity"][["method", "metrics.auroc"]]
            methods = sorted(box_df["method"].unique())
            vals = [box_df[box_df["method"] == m]["metrics.auroc"].values for m in methods]
            plt.boxplot(vals, labels=methods)
            plt.title("Seed Dispersion (AUROC, score=vacuity)")
            plt.ylabel("AUROC")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "seed_dispersion_auroc.png")
            plt.savefig(fig_dir / "seed_dispersion_auroc.pdf")
            plt.close()

    def save_appendix(self, df: pd.DataFrame) -> None:
        app_dir = self.out_dir / "appendix"
        app_dir.mkdir(parents=True, exist_ok=True)
        flat = pd.json_normalize(df.to_dict(orient="records"))
        lines = [
            "% Auto-generated experimental setup appendix",
            "\\section*{Experimental Setup}",
            f"Total records: {len(df)}\\\\",
            f"Methods: {', '.join(sorted(flat['method'].dropna().unique()))}\\\\",
            f"Method variants: {', '.join(sorted(flat.get('method_variant', pd.Series(dtype=str)).dropna().unique()))}\\\\",
            f"Datasets: {', '.join(sorted(flat['dataset'].dropna().unique()))}\\\\",
            f"Scores: {', '.join(sorted(flat.get('score_type', pd.Series(dtype=str)).dropna().unique()))}\\\\",
            f"Calibration: {', '.join(sorted(flat.get('calibration_type', pd.Series(dtype=str)).dropna().unique()))}\\\\",
        ]
        (app_dir / "exp_setup.tex").write_text("\n".join(lines), encoding="utf-8")

    def save_manifest(self) -> None:
        files = [str(p.relative_to(self.out_dir)) for p in self.out_dir.rglob("*") if p.is_file()]
        (self.out_dir / "manifest.json").write_text(json.dumps({"files": sorted(files)}, indent=2), encoding="utf-8")

    def build(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        df = self.collect()
        table = self.build_main_table(df)
        self.save_table(table)
        self.save_figures(df)
        self.save_appendix(df)
        self.save_manifest()
