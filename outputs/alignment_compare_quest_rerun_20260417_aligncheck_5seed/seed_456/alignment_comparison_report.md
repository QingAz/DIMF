# Alignment Comparison on Raw-Gap Lagged LiquidSugar

Raw dataset: `/gpfs/projects/p32954/dimf_liquid_sugar_repo-490/data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv`
Compared edge: `stage1_to_stage2`
Matched test samples: aligned=2612, noalign=2612

## Forecast Metrics

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| aligned | 7.303 | 9.941 | 0.154 |
| noalign | 6.152 | 8.654 | 0.359 |

## Lag Recovery

| model | subset | n | expected_lag_mae | argmax_acc | mean_entropy | mean_pred_expected |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | overall | 2612 | 0.282 | 0.980 | 0.322 | 0.241 |
| aligned | lagged_only | 48 | 2.434 | 0.000 | 0.346 | 0.232 |
| aligned | no_lag_only | 2564 | 0.242 | 0.998 | 0.321 | 0.242 |
| noalign | overall | 2612 | 0.049 | 0.982 | 0.000 | 0.000 |
| noalign | lagged_only | 48 | 2.667 | 0.000 | 0.000 | 0.000 |
| noalign | no_lag_only | 2564 | 0.000 | 1.000 | 0.000 | 0.000 |

## Per True Lag

| lag_gt | n | aligned_exp_mae | noalign_exp_mae | aligned_acc | noalign_acc | aligned_pred_mean | noalign_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 0.242 | 0.000 | 0.998 | 1.000 | 0.242 | 0.000 |
| 1 | 12 | 0.851 | 1.000 | 0.000 | 0.000 | 0.149 | 0.000 |
| 2 | 16 | 1.784 | 2.000 | 0.000 | 0.000 | 0.216 | 0.000 |
| 3 | 8 | 2.719 | 3.000 | 0.000 | 0.000 | 0.281 | 0.000 |
| 4 | 4 | 3.767 | 4.000 | 0.000 | 0.000 | 0.233 | 0.000 |
| 5 | 4 | 4.647 | 5.000 | 0.000 | 0.000 | 0.353 | 0.000 |
| 6 | 4 | 5.671 | 6.000 | 0.000 | 0.000 | 0.329 | 0.000 |

## Takeaways

- Lagged samples only: aligned expected-lag MAE 2.434 vs noalign 2.667.
- Lagged samples only: aligned argmax accuracy 0.000 vs noalign 0.000.
- Forecasting MAE: aligned 7.303 vs noalign 6.152.
- Overall mean predicted lag: aligned 0.241 vs noalign 0.000.

## Benchmark Metrics (4 items)

| metric | aligned | noalign |
| --- | --- | --- |
| Prediction MAE improvement (noalign - aligned) | -1.151 | NA |
| Prediction RMSE improvement (noalign - aligned) | -1.287 | NA |
| Block-in Expected-Lag MAE | 2.434 | 2.667 |
| Localization AUPRC | 0.021 | 0.000 |
| Localization best-F1 | 0.082 | 0.036 |
| Block-out False Alarm Rate (at best-F1 threshold) | 0.109 | 1.000 |

