from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.paper_artifacts import PaperArtifactsBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    builder = PaperArtifactsBuilder(runs_dir=Path(args.input), out_dir=Path(args.out))
    builder.build()
    print(f"Saved artifacts to {args.out}")


if __name__ == "__main__":
    main()
