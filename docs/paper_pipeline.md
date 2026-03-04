# Paper Pipeline

## Run Core Reproduction

```bash
python run.py preset core_repro
```

## Build Paper Artifacts

```bash
python scripts/paper/build_paper_artifacts.py --input runs --out artifacts/paper
```

## Outputs

- `tables/table_main.tex`
- `tables/table_main.csv`
- `figures/*.png, *.pdf`
- `appendix/exp_setup.tex`
- `manifest.json`
