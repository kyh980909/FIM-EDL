# Info-EDL Comparison: previous official baseline vs `beta=0.05`, `gamma=2.0` exact protocol

## Protocol

- Both rows use the official few-shot feature protocol
- `WideResNet28_10 + S2M2_R` features
- `novel` split
- `10,000` episodes
- official `INFO_EDL`

Important difference:
- previous baseline: `info_beta=1.0`, `info_gamma=1.0`, `Adam` fallback
- new run: `info_beta=0.05`, `info_gamma=2.0`, `LBFGS`

So this table is useful for inspection, but it is not a strict apples-to-apples hyperparameter comparison.

| Method | 5-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Info-EDL previous official rerun (`beta=1.0`, `gamma=1.0`, Adam) | 52.57 +/- 0.22 | 73.11 +/- 0.27 | 64.77 +/- 0.27 | 71.17 +/- 0.19 | 85.02 +/- 0.19 | 54.32 +/- 0.21 | 79.81 +/- 0.14 | 90.21 +/- 0.13 | 51.14 +/- 0.19 | 35.31 +/- 0.14 | 52.57 +/- 0.22 | 63.34 +/- 0.22 | 54.06 +/- 0.14 | 67.30 +/- 0.20 | 49.56 +/- 0.14 | 65.35 +/- 0.12 | 76.67 +/- 0.15 | 46.00 +/- 0.10 |
| Info-EDL exact (`beta=0.05`, `gamma=2.0`, LBFGS) | 45.65 +/- 0.25 | 70.31 +/- 0.29 | 58.54 +/- 0.17 | 57.37 +/- 0.26 | 78.55 +/- 0.25 | 58.39 +/- 0.15 | 61.89 +/- 0.24 | 81.13 +/- 0.23 | 58.22 +/- 0.15 | 30.99 +/- 0.16 | 50.25 +/- 0.24 | 54.37 +/- 0.10 | 41.48 +/- 0.18 | 58.84 +/- 0.24 | 54.26 +/- 0.10 | 46.24 +/- 0.17 | 62.60 +/- 0.23 | 54.12 +/- 0.09 |

## Delta vs previous rerun

- `5-way 1-shot`: Acc `-6.92`, Conf `-2.80`, OOD `-6.23`
- `5-way 5-shot`: Acc `-13.80`, Conf `-6.47`, OOD `+4.07`
- `5-way 20-shot`: Acc `-17.92`, Conf `-9.08`, OOD `+7.08`
- `10-way 1-shot`: Acc `-4.32`, Conf `-2.32`, OOD `-8.97`
- `10-way 5-shot`: Acc `-12.58`, Conf `-8.46`, OOD `+4.70`
- `10-way 20-shot`: Acc `-19.11`, Conf `-14.07`, OOD `+8.12`

## Sources

- Previous rerun summary: [/home/yongho/FIM-EDL/results/fewshot_official_v5/summary_mean_std.csv](/home/yongho/FIM-EDL/results/fewshot_official_v5/summary_mean_std.csv)
- Exact `beta=0.05`, `gamma=2.0` summary: [/home/yongho/FIM-EDL/results/fewshot_official_beta005_gamma20_exact/summary_mean_std.csv](/home/yongho/FIM-EDL/results/fewshot_official_beta005_gamma20_exact/summary_mean_std.csv)
- Exact protocol table: [/home/yongho/FIM-EDL/results/paper_tables/iedl_table4_beta005_gamma20_exact/table4_iedl_style.md](/home/yongho/FIM-EDL/results/paper_tables/iedl_table4_beta005_gamma20_exact/table4_iedl_style.md)
