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
        grp = flat.groupby(["method", "dataset"])
        table = grp[["metrics.auroc", "metrics.fpr95", "metrics.accuracy"]].agg(["mean", "std"])
        return table

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

        plt.figure(figsize=(6, 4))
        for method, sub in flat.groupby("method"):
            sub = sub.groupby("dataset")["metrics.auroc"].mean().reset_index()
            plt.plot(sub["dataset"], sub["metrics.auroc"], marker="o", label=method)
        plt.title("OOD AUROC by Method")
        plt.ylabel("AUROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "ood_roc_placeholder.png")
        plt.savefig(fig_dir / "ood_roc_placeholder.pdf")
        plt.close()

    def save_appendix(self, df: pd.DataFrame) -> None:
        app_dir = self.out_dir / "appendix"
        app_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "% Auto-generated experimental setup appendix",
            "\\section*{Experimental Setup}",
            f"Total records: {len(df)}\\\\",
            f"Methods: {', '.join(sorted(df['method'].unique()))}\\\\",
            f"Datasets: {', '.join(sorted(df['dataset'].unique()))}\\\\",
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
