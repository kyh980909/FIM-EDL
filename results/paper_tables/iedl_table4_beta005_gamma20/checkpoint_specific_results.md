# Info-EDL Few-Shot Results for `beta=0.05`, `gamma=2.0`

Checkpoint: `/home/yongho/FIM-EDL/runs/info_edl/seed_0/20260317T131319838373Z/checkpoints/best.ckpt`

Protocol notes:
- Local checkpoint-based few-shot evaluation on `miniimagenet` with `eval_split=test`
- `episodes=20`, `adapt_steps=10`, `adapt_lr=0.01`
- `20-shot` is unavailable on this local split because some test classes do not have enough images for support/query sampling

| Method | 5-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 5-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 1-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 5-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) | 10-Way 20-Shot Acc. | Conf. (Max.alpha) | OOD (alpha0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Info-EDL (`beta=0.05`, `gamma=2.0`) | 37.00 +/- 9.71 | 70.23 +/- 14.37 | 50.24 +/- 4.27 | 45.40 +/- 9.24 | 61.23 +/- 11.37 | 44.83 +/- 5.79 | N/A | N/A | N/A | 25.50 +/- 5.79 | 63.06 +/- 11.22 | 52.25 +/- 4.44 | 38.70 +/- 4.40 | 63.96 +/- 7.43 | 44.15 +/- 2.22 | N/A | N/A | N/A |
