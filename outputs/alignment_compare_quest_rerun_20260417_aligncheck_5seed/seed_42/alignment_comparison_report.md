# Alignment Comparison on Raw-Gap Lagged LiquidSugar

Raw dataset: `/gpfs/projects/p32954/dimf_liquid_sugar_repo-490/data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv`
Compared edge: `stage1_to_stage2`
Matched test samples: aligned=2612, noalign=2612

## Forecast Metrics

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| aligned | 6.230 | 8.800 | 0.337 |
| noalign | 5.757 | 8.284 | 0.413 |

## Lag Recovery

| model | subset | n | expected_lag_mae | argmax_acc | mean_entropy | mean_pred_expected |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | overall | 2612 | 1.075 | 0.967 | 0.990 | 1.060 |
| aligned | lagged_only | 48 | 1.773 | 0.000 | 0.987 | 0.965 |
| aligned | no_lag_only | 2564 | 1.062 | 0.985 | 0.990 | 1.062 |
| noalign | overall | 2612 | 0.049 | 0.982 | 0.000 | 0.000 |
| noalign | lagged_only | 48 | 2.667 | 0.000 | 0.000 | 0.000 |
| noalign | no_lag_only | 2564 | 0.000 | 1.000 | 0.000 | 0.000 |

## Per True Lag

| lag_gt | n | aligned_exp_mae | noalign_exp_mae | aligned_acc | noalign_acc | aligned_pred_mean | noalign_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 1.062 | 0.000 | 0.985 | 1.000 | 1.062 | 0.000 |
| 1 | 12 | 0.424 | 1.000 | 0.000 | 0.000 | 0.801 | 0.000 |
| 2 | 16 | 1.050 | 2.000 | 0.000 | 0.000 | 0.994 | 0.000 |
| 3 | 8 | 1.960 | 3.000 | 0.000 | 0.000 | 1.040 | 0.000 |
| 4 | 4 | 3.053 | 4.000 | 0.000 | 0.000 | 0.947 | 0.000 |
| 5 | 4 | 3.944 | 5.000 | 0.000 | 0.000 | 1.056 | 0.000 |
| 6 | 4 | 4.884 | 6.000 | 0.000 | 0.000 | 1.116 | 0.000 |

## Takeaways

- Lagged samples only: aligned expected-lag MAE 1.773 vs noalign 2.667.
- Lagged samples only: aligned argmax accuracy 0.000 vs noalign 0.000.
- Forecasting MAE: aligned 6.230 vs noalign 5.757.
- Overall mean predicted lag: aligned 1.060 vs noalign 0.000.

## Benchmark Metrics (4 items)

| metric | aligned | noalign |
| --- | --- | --- |
| Prediction MAE improvement (noalign - aligned) | -0.472 | NA |
| Prediction RMSE improvement (noalign - aligned) | -0.516 | NA |
| Block-in Expected-Lag MAE | 1.773 | 2.667 |
| Localization AUPRC | 0.022 | 0.000 |
| Localization best-F1 | 0.077 | 0.036 |
| Block-out False Alarm Rate (at best-F1 threshold) | 0.039 | 1.000 |

