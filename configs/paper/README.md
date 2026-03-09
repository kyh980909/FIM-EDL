# Paper Table Templates

## I-EDL Table 2 Style

Reference baseline values can be filled in:

- `configs/paper/iedl_table2_reference_template.csv`

Generate the combined table with project results:

```bash
python3 scripts/paper/build_iedl_table2.py \
  --summary-csv results/eval/summary_mean_std.csv \
  --reference-csv configs/paper/iedl_table2_reference_template.csv \
  --out-dir results/paper_tables/iedl_table2
```

Outputs:

- `results/paper_tables/iedl_table2/table2_iedl_style.csv`
- `results/paper_tables/iedl_table2/table2_iedl_style.md`
- `results/paper_tables/iedl_table2/table2_iedl_style.tex`

## I-EDL Table 3 Style

Reference baseline values are stored in:

- `configs/paper/iedl_table3_reference_template.csv`

Generate the CIFAR10 misclassification-detection summary with project results:

```bash
python3 scripts/paper/build_iedl_table3.py \
  --runs runs \
  --dataset cifar10 \
  --reference-csv configs/paper/iedl_table3_reference_template.csv \
  --out-dir results/paper_tables/iedl_table3
```

Outputs:

- `results/paper_tables/iedl_table3/table3_iedl_style.csv`
- `results/paper_tables/iedl_table3/table3_iedl_style.md`
- `results/paper_tables/iedl_table3/table3_iedl_style.tex`
