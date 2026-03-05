# Info-EDL Reproduction Platform

Hydra + PyTorch Lightning 기반 Info-EDL 재현/확장 플랫폼입니다.

## Quick Start

```bash
pip install -e .
python run.py preset quick_smoke
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
```

For mini-ImageNet/CUB, place image folders under:
- `data/miniimagenet/{train,val,test}/<class>/*.jpg`
- `data/cub/test/<class>/*.jpg`

Paper-aligned backbones:
- `vgg16` for CIFAR10 setting
- `convnet` for MNIST setting
- `wrn28_10` for few-shot setting

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
```
