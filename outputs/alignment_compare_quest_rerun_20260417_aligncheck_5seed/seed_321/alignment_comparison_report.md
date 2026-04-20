# Alignment Comparison on Raw-Gap Lagged LiquidSugar

Raw dataset: `/gpfs/projects/p32954/dimf_liquid_sugar_repo-490/data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv`
Compared edge: `stage1_to_stage2`
Matched test samples: aligned=2612, noalign=2612

## Forecast Metrics

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| aligned | 5.720 | 8.319 | 0.408 |
| noalign | 6.012 | 8.552 | 0.374 |

## Lag Recovery

| model | subset | n | expected_lag_mae | argmax_acc | mean_entropy | mean_pred_expected |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | overall | 2612 | 3.731 | 0.371 | 2.252 | 3.764 |
| aligned | lagged_only | 48 | 1.671 | 0.250 | 2.151 | 3.429 |
| aligned | no_lag_only | 2564 | 3.770 | 0.374 | 2.254 | 3.770 |
| noalign | overall | 2612 | 0.049 | 0.982 | 0.000 | 0.000 |
| noalign | lagged_only | 48 | 2.667 | 0.000 | 0.000 | 0.000 |
| noalign | no_lag_only | 2564 | 0.000 | 1.000 | 0.000 | 0.000 |

## Per True Lag

| lag_gt | n | aligned_exp_mae | noalign_exp_mae | aligned_acc | noalign_acc | aligned_pred_mean | noalign_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 3.770 | 0.000 | 0.374 | 1.000 | 3.770 | 0.000 |
| 1 | 12 | 2.660 | 1.000 | 0.000 | 0.000 | 3.660 | 0.000 |
| 2 | 16 | 1.523 | 2.000 | 0.750 | 0.000 | 3.523 | 0.000 |
| 3 | 8 | 0.302 | 3.000 | 0.000 | 0.000 | 3.225 | 0.000 |
| 4 | 4 | 0.635 | 4.000 | 0.000 | 0.000 | 3.365 | 0.000 |
| 5 | 4 | 1.879 | 5.000 | 0.000 | 0.000 | 3.121 | 0.000 |
| 6 | 4 | 2.857 | 6.000 | 0.000 | 0.000 | 3.143 | 0.000 |

## Takeaways

- Lagged samples only: aligned expected-lag MAE 1.671 vs noalign 2.667.
- Lagged samples only: aligned argmax accuracy 0.250 vs noalign 0.000.
- Forecasting MAE: aligned 5.720 vs noalign 6.012.
- Overall mean predicted lag: aligned 3.764 vs noalign 0.000.

## Benchmark Metrics (4 items)

| metric | aligned | noalign |
| --- | --- | --- |
| Prediction MAE improvement (noalign - aligned) | 0.292 | NA |
| Prediction RMSE improvement (noalign - aligned) | 0.233 | NA |
| Block-in Expected-Lag MAE | 1.671 | 2.667 |
| Localization AUPRC | 0.028 | 0.000 |
| Localization best-F1 | 0.087 | 0.036 |
| Block-out False Alarm Rate (at best-F1 threshold) | 0.101 | 1.000 |

