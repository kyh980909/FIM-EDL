# Info-EDL Reproduction Platform

Hydra + PyTorch Lightning 기반 Info-EDL 재현/확장 플랫폼입니다.

## Quick Start

```bash
uv sync --extra dev
source .venv/bin/activate
python run.py preset quick_smoke
```

If you need a legacy editable install instead of `uv`, use:

```bash
pip install -e ".[dev]"
```

## Hyperparameter Baseline

Default training hyperparameters are aligned with the Fisher-EDL CIFAR setup
from `https://github.com/kyh980909/Fisher-EDL`:
- `epochs=100`
- `optimizer=adam`
- `lr=1e-3`
- `weight_decay=0.0`
- `batch_size=128`
- `num_workers=2`

## Core Repro

```bash
python run.py preset core_repro
```

`core_repro` now includes 5 methods:
- `edl_l1`
- `edl_l01`
- `edl_l0001`
- `info_edl`
- `iedl_ref`

## Pilot/Confirm Presets

```bash
# CIFAR-10 OOD pilot (seed 0,1)
python run.py preset pilot_full

# CIFAR-10 OOD confirm (selected methods, seed 0..4)
python run.py preset confirm_full

# MNIST OOD pilot/confirm
python run.py preset pilot_mnist
python run.py preset confirm_mnist

# mini-ImageNet/CUB folder-based presets
python run.py preset pilot_fewshot
python run.py preset confirm_fewshot

# I-EDL paper backbone presets
python run.py preset paper_cifar_pilot
python run.py preset paper_mnist_pilot
python run.py preset paper_fewshot_pilot

# I-EDL ref.pdf-aligned presets for Tables 2 and 3
python run.py preset paper_iedl_mnist_ref
python run.py preset paper_iedl_cifar_ref
```

For mini-ImageNet/CUB, place image folders under:
- `data/miniimagenet/{train,val,test}/<class>/*.jpg`
- `data/cub/test/<class>/*.jpg`

Paper-aligned backbones:
- `vgg16` for CIFAR10 setting
- `convnet` for MNIST setting
- `wrn28_10` for few-shot setting

`paper_iedl_mnist_ref` and `paper_iedl_cifar_ref` match the Appendix C.2 training setup from `ref.pdf` more closely:
- `epochs=200`
- EDL KL weight annealing uses the paper schedule `lambda_t = min(1, t/T)`, so `loss.anneal_epochs=200`
- `batch_size=64`
- `lr=1e-3` for MNIST
- `lr=5e-4` for CIFAR10
- `convnet` on MNIST, `vgg16` on CIFAR10

## Override Examples

```bash
python run.py preset core_repro loss.beta=0.8 loss.gamma=1.2
python -m src.train experiment=info_edl seed=0 trainer.max_epochs=200
python -m src.eval experiment=info_edl checkpoint=/path/to/ckpt.ckpt
```

## Paper Artifacts

```bash
python scripts/paper/export_eval_results.py --runs runs --out results/eval
python scripts/paper/build_paper_artifacts.py --input runs --out artifacts/paper
python3 scripts/paper/build_iedl_table2.py --summary-csv results/eval/summary_mean_std.csv --reference-csv configs/paper/iedl_table2_reference_template.csv --out-dir results/paper_tables/iedl_table2
python3 scripts/paper/build_iedl_table3.py --runs runs --dataset cifar10 --reference-csv configs/paper/iedl_table3_reference_template.csv --out-dir results/paper_tables/iedl_table3
```
