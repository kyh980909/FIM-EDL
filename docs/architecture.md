# Architecture

## Extension Rules

- Add backbone/head/loss/score as a new file.
- Register once with `@..._REGISTRY.register("name")`.
- Select component from Hydra config without code branching.

## Contracts

- `BackboneProtocol`: forward(x) and `out_dim`.
- `HeadProtocol`: returns `logits/evidence/alpha/probs`.
- `LossProtocol`: returns `total/fit/reg/aux/schema_version`.
- `ScoreProtocol`: returns uncertainty score tensor.

## Dataset Adapter Contract

- `num_classes()`
- `class_names()`
- `normalization_spec()`
- `id_dataloaders()`
- `ood_dataloaders()`
