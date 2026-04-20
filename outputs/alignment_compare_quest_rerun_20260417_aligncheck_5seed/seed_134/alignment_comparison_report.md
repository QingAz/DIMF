# Alignment Comparison on Raw-Gap Lagged LiquidSugar

Raw dataset: `/gpfs/projects/p32954/dimf_liquid_sugar_repo-490/data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv`
Compared edge: `stage1_to_stage2`
Matched test samples: aligned=2612, noalign=2612

## Forecast Metrics

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| aligned | 6.388 | 9.047 | 0.299 |
| noalign | 6.449 | 9.076 | 0.295 |

## Lag Recovery

| model | subset | n | expected_lag_mae | argmax_acc | mean_entropy | mean_pred_expected |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | overall | 2612 | 1.155 | 0.949 | 0.951 | 1.147 |
| aligned | lagged_only | 48 | 1.657 | 0.021 | 1.051 | 1.233 |
| aligned | no_lag_only | 2564 | 1.145 | 0.967 | 0.949 | 1.145 |
| noalign | overall | 2612 | 0.049 | 0.982 | 0.000 | 0.000 |
| noalign | lagged_only | 48 | 2.667 | 0.000 | 0.000 | 0.000 |
| noalign | no_lag_only | 2564 | 0.000 | 1.000 | 0.000 | 0.000 |

## Per True Lag

| lag_gt | n | aligned_exp_mae | noalign_exp_mae | aligned_acc | noalign_acc | aligned_pred_mean | noalign_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 1.145 | 0.000 | 0.967 | 1.000 | 1.145 | 0.000 |
| 1 | 12 | 0.470 | 1.000 | 0.000 | 0.000 | 1.145 | 0.000 |
| 2 | 16 | 0.992 | 2.000 | 0.062 | 0.000 | 1.216 | 0.000 |
| 3 | 8 | 1.625 | 3.000 | 0.000 | 0.000 | 1.375 | 0.000 |
| 4 | 4 | 2.374 | 4.000 | 0.000 | 0.000 | 1.626 | 0.000 |
| 5 | 4 | 3.943 | 5.000 | 0.000 | 0.000 | 1.057 | 0.000 |
| 6 | 4 | 4.943 | 6.000 | 0.000 | 0.000 | 1.057 | 0.000 |

## Takeaways

- Lagged samples only: aligned expected-lag MAE 1.657 vs noalign 2.667.
- Lagged samples only: aligned argmax accuracy 0.021 vs noalign 0.000.
- Forecasting MAE: aligned 6.388 vs noalign 6.449.
- Overall mean predicted lag: aligned 1.147 vs noalign 0.000.

## Benchmark Metrics (4 items)

| metric | aligned | noalign |
| --- | --- | --- |
| Prediction MAE improvement (noalign - aligned) | 0.061 | NA |
| Prediction RMSE improvement (noalign - aligned) | 0.030 | NA |
| Block-in Expected-Lag MAE | 1.657 | 2.667 |
| Localization AUPRC | 0.024 | 0.000 |
| Localization best-F1 | 0.084 | 0.036 |
| Block-out False Alarm Rate (at best-F1 threshold) | 0.035 | 1.000 |

