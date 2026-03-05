from __future__ import annotations

import argparse
import json
from pathlib import Path


def migrate_file(path: Path, target_version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        row = json.loads(line)
        row["results_schema_version"] = target_version
        row.setdefault("method_variant", row.get("method", ""))
        row.setdefault("score_type", "vacuity")
        row.setdefault("calibration_type", "none")
        out.append(json.dumps(row))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate metrics schema version")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--to", required=True, help="Target schema version, e.g., v2")
    args = parser.parse_args()

    root = Path(args.runs)
    for path in root.rglob("metrics.jsonl"):
        migrate_file(path, args.to)
        print(f"Migrated {path} -> {args.to}")


if __name__ == "__main__":
    main()
