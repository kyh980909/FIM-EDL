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

## Build I-EDL Table 2 Style Summary

```bash
python3 scripts/paper/build_iedl_table2.py \
  --summary-csv results/eval/summary_mean_std.csv \
  --reference-csv configs/paper/iedl_table2_reference_template.csv \
  --out-dir results/paper_tables/iedl_table2
```

This script:

- keeps the I-EDL Table 2 layout
- fills project rows from `results/eval/summary_mean_std.csv`
- keeps literature baselines in a separate template CSV for manual entry
- exports `csv`, `md`, and `tex`

## Build I-EDL Table 3 Style Summary

```bash
python3 scripts/paper/build_iedl_table3.py \
  --runs runs \
  --dataset cifar10 \
  --reference-csv configs/paper/iedl_table3_reference_template.csv \
  --out-dir results/paper_tables/iedl_table3
```

This script:

- reads `conf_eval` rows emitted by `src.eval`
- aggregates CIFAR10 misclassification-detection AUPR over seeds
- keeps literature baselines in a separate template CSV
- exports `csv`, `md`, and `tex`

## Status of Tables 4 and 5

Table 4 and Table 5 in `ref.pdf` require a dedicated few-shot episodic evaluation loop:

- frozen pre-trained `wrn28_10` features
- `N`-way `K`-shot episode sampling on mini-ImageNet meta-test
- per-episode adaptation of a 1-layer classifier
- matched CUB query sampling for OOD AUPR

The current Lightning train/eval pipeline is batch-supervised and does not yet implement that episodic protocol.
