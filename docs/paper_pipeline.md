# Paper Pipeline

## Run Core Reproduction

```bash
uv run python run.py preset core_repro
```

## Build Paper Artifacts

```bash
uv run python scripts/paper/build_paper_artifacts.py --input runs --out artifacts/paper
```

## Outputs

- `tables/table_main.tex`
- `tables/table_main.csv`
- `figures/*.png, *.pdf`
- `appendix/exp_setup.tex`
- `manifest.json`

## Build I-EDL Table 2 Style Summary

```bash
uv run python scripts/paper/build_iedl_table2.py \
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
uv run python scripts/paper/build_iedl_table3.py \
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

Few-shot episodic evaluation is now available through `src.eval_fewshot` and
`scripts/paper/export_fewshot_results.py`.

Implemented protocol:

- frozen checkpoint backbone as the feature extractor
- `N`-way `K`-shot episodic sampling on mini-ImageNet
- per-episode adaptation of a 1-layer evidential classifier
- matched CUB query sampling for OOD AUPR/AUROC/FPR95

Current gaps versus the exact `ref.pdf` setup:

- the repo does not bundle the external pre-trained WideResNet from Yang et al. (2021)
- local mini-ImageNet folders may have fewer images per class than the paper protocol, so query count is capped by availability

## Run Official Few-Shot Protocol

The paper-aligned rerun path uses the official `code_fsl` implementation with the
released `WideResNet28_10_S2M2_R` feature bundle.

Launch a 5-way Info-EDL rerun:

```bash
screen -dmS infoedl_5w_official \
  env METHOD=infoedl WAYS=5 TASKS=10000 TORCH_THREADS=1 \
  LOG_PATH=outputs/logs/infoedl_5w_official.log \
  RESULTS_DIR=results/fewshot_official/raw \
  bash scripts/paper/run_info_edl_official_fewshot.sh
```

The launcher defaults to conservative thread settings:
`TORCH_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
Keep those values unless you have confirmed your local BLAS stack is stable.

For stability, the launcher defaults to `OPTIMIZER_NAME=adam`.
On some recent PyTorch builds, the official `LBFGS` path crashes inside
`torch.optim.lbfgs` with `Floating point exception`. The fallback keeps the
same episodic adaptation setup and support/query protocol, but changes the
inner-loop optimizer.

If your environment has a stable `LBFGS`, you can switch back with
`OPTIMIZER_NAME=lbfgs LBFGS_LR=0.25 LBFGS_LINE_SEARCH_FN=none`.

If a full `10000`-task run is still unstable on your machine, run it in chunks.
The launcher supports `TASK_START`, so `1000 x 10` chunks will cover the same
task ids without repeating earlier episodes.

Aggregate official rerun CSVs:

```bash
uv run python scripts/paper/export_official_fewshot_results.py \
  --results-dir results/fewshot_official/raw \
  --out results/fewshot_official
```

Merge those rows into the Table 4 builder:

```bash
uv run python scripts/paper/build_iedl_table4.py \
  --runs runs \
  --methods edl_official,iedl_official,info_edl_official,info_edl \
  --reference-csv configs/paper/iedl_table4_reference_template.csv \
  --official-summary-csv results/fewshot_official/summary_mean_std.csv \
  --out-dir results/paper_tables/iedl_table4
```
