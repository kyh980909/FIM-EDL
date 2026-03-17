# Info-EDL Comparison: official rerun vs `beta=0.05`, `gamma=2.0`

## Protocol

- `Info-EDL (official rerun)`: official feature protocol, `10,000` episodes, `novel` split
- `Info-EDL (beta=0.05, gamma=2.0)`: local checkpoint-based protocol, `20` episodes, `test` split
- These are not perfectly apples-to-apples. The table is for direct inspection, not a paper-quality final comparison.

| Method | 5-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Info-EDL (official rerun) | 52.57 +/- 0.22 | 73.11 +/- 0.27 | 64.77 +/- 0.27 | 71.17 +/- 0.19 | 85.02 +/- 0.19 | 54.32 +/- 0.21 | 79.81 +/- 0.14 | 90.21 +/- 0.13 | 51.14 +/- 0.19 | 35.31 +/- 0.14 | 52.57 +/- 0.22 | 63.34 +/- 0.22 | 54.06 +/- 0.14 | 67.30 +/- 0.20 | 49.56 +/- 0.14 | 65.35 +/- 0.12 | 76.67 +/- 0.15 | 46.00 +/- 0.10 |
| Info-EDL (`beta=0.05`, `gamma=2.0`) | 37.00 +/- 9.71 | 70.23 +/- 14.37 | 50.24 +/- 4.27 | 45.40 +/- 3.93 | 61.23 +/- 5.65 | 44.83 +/- 2.61 | N/A | N/A | N/A | 25.50 +/- 5.79 | 63.06 +/- 11.22 | 52.25 +/- 4.44 | 38.70 +/- 4.40 | 63.96 +/- 7.43 | 44.15 +/- 2.22 | N/A | N/A | N/A |

## Delta vs official rerun

- `5-way 1-shot`: Acc `-15.57`, Conf `-2.88`, OOD `-14.53`
- `5-way 5-shot`: Acc `-25.77`, Conf `-23.79`, OOD `-9.49`
- `10-way 1-shot`: Acc `-9.81`, Conf `+10.49`, OOD `-11.09`
- `10-way 5-shot`: Acc `-15.36`, Conf `-3.34`, OOD `-5.41`

## Sources

- Official table: [/home/yongho/FIM-EDL/results/paper_tables/iedl_table4_official_v5/table4_iedl_style.md](/home/yongho/FIM-EDL/results/paper_tables/iedl_table4_official_v5/table4_iedl_style.md)
- Official summary: [/home/yongho/FIM-EDL/results/fewshot_official_v5/summary_mean_std.csv](/home/yongho/FIM-EDL/results/fewshot_official_v5/summary_mean_std.csv)
- New checkpoint summary: [/home/yongho/FIM-EDL/results/fewshot_beta005_gamma20/summary_mean_std.csv](/home/yongho/FIM-EDL/results/fewshot_beta005_gamma20/summary_mean_std.csv)
