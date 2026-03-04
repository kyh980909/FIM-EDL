from __future__ import annotations

import argparse
from pathlib import Path

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
