# Info-EDL Exact Comparison: `beta=1.0`, `gamma=1.0` vs `beta=0.05`, `gamma=2.0`

## Protocol

- Both rows use the same official few-shot protocol
- `WideResNet28_10 + S2M2_R` features
- `novel` split
- `10,000` episodes
- official `INFO_EDL`
- `LBFGS`, `lbfgs_iters=100`

This is the strict apples-to-apples hyperparameter comparison.

| Method | 5-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Info-EDL exact baseline (`beta=1.0`, `gamma=1.0`) | 45.82 +/- 0.25 | 70.63 +/- 0.29 | 58.65 +/- 0.17 | 57.51 +/- 0.26 | 78.86 +/- 0.25 | 58.49 +/- 0.15 | 62.04 +/- 0.24 | 81.44 +/- 0.23 | 58.32 +/- 0.15 | 30.99 +/- 0.16 | 50.27 +/- 0.24 | 54.37 +/- 0.10 | 41.48 +/- 0.18 | 58.86 +/- 0.24 | 54.26 +/- 0.10 | 46.25 +/- 0.17 | 62.62 +/- 0.23 | 54.12 +/- 0.09 |
| Info-EDL exact (`beta=0.05`, `gamma=2.0`) | 45.65 +/- 0.25 | 70.31 +/- 0.29 | 58.54 +/- 0.17 | 57.37 +/- 0.26 | 78.55 +/- 0.25 | 58.39 +/- 0.15 | 61.89 +/- 0.24 | 81.13 +/- 0.23 | 58.22 +/- 0.15 | 30.99 +/- 0.16 | 50.25 +/- 0.24 | 54.37 +/- 0.10 | 41.48 +/- 0.18 | 58.84 +/- 0.24 | 54.26 +/- 0.10 | 46.24 +/- 0.17 | 62.60 +/- 0.23 | 54.12 +/- 0.09 |

## Delta vs exact baseline

- `5-way 1-shot`: Acc `-0.17`, Conf `-0.32`, OOD `-0.11`
- `5-way 5-shot`: Acc `-0.14`, Conf `-0.31`, OOD `-0.10`
- `5-way 20-shot`: Acc `-0.15`, Conf `-0.31`, OOD `-0.10`
- `10-way 1-shot`: Acc `0.00`, Conf `-0.02`, OOD `0.00`
- `10-way 5-shot`: Acc `0.00`, Conf `-0.02`, OOD `0.00`
- `10-way 20-shot`: Acc `-0.01`, Conf `-0.02`, OOD `0.00`

## Sources

- Exact baseline summary: [/home/yongho/FIM-EDL/results/fewshot_official_beta1_gamma1_exact/summary_mean_std.csv](/home/yongho/FIM-EDL/results/fewshot_official_beta1_gamma1_exact/summary_mean_std.csv)
- Exact `beta=0.05`, `gamma=2.0` summary: [/home/yongho/FIM-EDL/results/fewshot_official_beta005_gamma20_exact/summary_mean_std.csv](/home/yongho/FIM-EDL/results/fewshot_official_beta005_gamma20_exact/summary_mean_std.csv)
