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

## Override Examples

```bash
python run.py preset core_repro loss.beta=0.8 loss.gamma=1.2
python -m src.train experiment=info_edl seed=0 trainer.max_epochs=200
python -m src.eval experiment=info_edl checkpoint=/path/to/ckpt.ckpt
```

## Paper Artifacts

```bash
python scripts/paper/build_paper_artifacts.py --input runs --out artifacts/paper
```
