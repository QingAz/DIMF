# Alignment Comparison on Raw-Gap Lagged LiquidSugar

Raw dataset: `/gpfs/projects/p32954/dimf_liquid_sugar_repo-490/data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv`
Compared edge: `stage1_to_stage2`
Matched test samples: aligned=2612, noalign=2612

## Forecast Metrics

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| aligned | 6.266 | 8.930 | 0.317 |
| noalign | 6.159 | 8.746 | 0.345 |

## Lag Recovery

| model | subset | n | expected_lag_mae | argmax_acc | mean_entropy | mean_pred_expected |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | overall | 2612 | 3.586 | 0.759 | 2.235 | 3.626 |
| aligned | lagged_only | 48 | 1.640 | 0.125 | 2.318 | 3.842 |
| aligned | no_lag_only | 2564 | 3.622 | 0.771 | 2.234 | 3.622 |
| noalign | overall | 2612 | 0.049 | 0.982 | 0.000 | 0.000 |
| noalign | lagged_only | 48 | 2.667 | 0.000 | 0.000 | 0.000 |
| noalign | no_lag_only | 2564 | 0.000 | 1.000 | 0.000 | 0.000 |

## Per True Lag

| lag_gt | n | aligned_exp_mae | noalign_exp_mae | aligned_acc | noalign_acc | aligned_pred_mean | noalign_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 3.622 | 0.000 | 0.771 | 1.000 | 3.622 | 0.000 |
| 1 | 12 | 2.493 | 1.000 | 0.000 | 0.000 | 3.493 | 0.000 |
| 2 | 16 | 1.851 | 2.000 | 0.375 | 0.000 | 3.851 | 0.000 |
| 3 | 8 | 0.957 | 3.000 | 0.000 | 0.000 | 3.957 | 0.000 |
| 4 | 4 | 0.459 | 4.000 | 0.000 | 0.000 | 3.735 | 0.000 |
| 5 | 4 | 0.805 | 5.000 | 0.000 | 0.000 | 4.195 | 0.000 |
| 6 | 4 | 1.623 | 6.000 | 0.000 | 0.000 | 4.377 | 0.000 |

## Takeaways

- Lagged samples only: aligned expected-lag MAE 1.640 vs noalign 2.667.
- Lagged samples only: aligned argmax accuracy 0.125 vs noalign 0.000.
- Forecasting MAE: aligned 6.266 vs noalign 6.159.
- Overall mean predicted lag: aligned 3.626 vs noalign 0.000.

## Benchmark Metrics (4 items)

| metric | aligned | noalign |
| --- | --- | --- |
| Prediction MAE improvement (noalign - aligned) | -0.107 | NA |
| Prediction RMSE improvement (noalign - aligned) | -0.184 | NA |
| Block-in Expected-Lag MAE | 1.640 | 2.667 |
| Localization AUPRC | 0.022 | 0.000 |
| Localization best-F1 | 0.053 | 0.036 |
| Block-out False Alarm Rate (at best-F1 threshold) | 0.324 | 1.000 |

