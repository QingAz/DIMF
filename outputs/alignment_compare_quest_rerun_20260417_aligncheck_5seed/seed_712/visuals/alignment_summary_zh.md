# Raw-Gap Alignment 结果汇总

## 结论

1. 预测 `yield_flow` 时，alignment 打开后 `MAE` 从 `6.159` 升到 `6.266`，这一轮预测指标没有收益。
2. 只看真正加了 lag 的样本，alignment 的 `expected lag MAE` 从 `2.667` 降到 `1.640`，说明它确实在恢复 lag 大小。
3. `argmax lag accuracy` 仍然是 `0`；在 lagged-only 样本上，aligned 的 `argmax lag` 均值为 `0.542`（众数 `0`），noalign 为 `0.000`（众数 `0`）；aligned 的 `expected lag` 均值为 `3.842`。

## 指标定义

- expected lag：模型预测分布 `π(ℓ|t)` 的期望值 `sum(ℓ·π(ℓ|t))`，对应 `pred_expected_lag`。
- argmax lag：`argmax_ℓ π(ℓ|t)`，对应 `pred_argmax_lag`。
- thresholded lag / nonzero score：`P(ℓ>0|t)=1-π(0|t)`，用于 PR / best‑F1 / FAR。
- mean predicted lag：在指定子集上对 `pred_expected_lag` 求均值。
- lagged‑only：只统计 `lag_gt > 0` 的样本；no‑lag‑only：`lag_gt == 0`；overall：全体样本。

## 图表

- Forecast 指标：[forecast_metrics.png](./forecast_metrics.png)
- Lag 子集对比：[lag_subset_metrics.png](./lag_subset_metrics.png)
- 各真实 lag 的误差：[lag_mae_by_true_lag.png](./lag_mae_by_true_lag.png)
- 各 lag block 时间轴：[lag_block_panels.png](./lag_block_panels.png)

## 关键数字

| 指标 | aligned | noalign |
| --- | --- | --- |
| Forecast MAE | 6.266 | 6.159 |
| Forecast RMSE | 8.930 | 8.746 |
| Forecast R2 | 0.317 | 0.345 |
| Lagged-only expected lag MAE | 1.640 | 2.667 |
| Lagged-only argmax acc | 0.125 | 0.000 |
| Mean predicted lag on lagged samples | 3.842 | 0.000 |

## 分真实 lag

| true lag | n | aligned MAE | noalign MAE | aligned pred mean | noalign pred mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 3.622 | 0.000 | 3.622 | 0.000 |
| 1 | 12 | 2.493 | 1.000 | 3.493 | 0.000 |
| 2 | 16 | 1.851 | 2.000 | 3.851 | 0.000 |
| 3 | 8 | 0.957 | 3.000 | 3.957 | 0.000 |
| 4 | 4 | 0.459 | 4.000 | 3.735 | 0.000 |
| 5 | 4 | 0.805 | 5.000 | 4.195 | 0.000 |
| 6 | 4 | 1.623 | 6.000 | 4.377 | 0.000 |

## Lag Block 摘要

| block | start | end | true lag | aligned pred mean | noalign pred mean | aligned MAE | noalign MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-03-25 07:15 | 2022-03-25 09:00 | 1 | 3.164 | 0.000 | 1.664 | 1.500 |
| 2 | 2022-04-07 12:45 | 2022-04-07 14:30 | 2 | 4.631 | 0.000 | 1.400 | 4.000 |
| 3 | 2022-04-13 01:45 | 2022-04-13 03:30 | 2 | 4.001 | 0.000 | 1.659 | 4.000 |
| 4 | 2022-04-18 17:30 | 2022-04-18 19:15 | 1 | 4.074 | 0.000 | 1.595 | 2.500 |
| 5 | 2022-04-27 16:45 | 2022-04-27 18:30 | 1 | 3.500 | 0.000 | 1.341 | 2.500 |
| 6 | 2022-04-29 10:15 | 2022-04-29 12:00 | 1 | 3.682 | 0.000 | 2.182 | 1.500 |

## Conditional Bias Table

按 true lag 分组的 `E[\hat d | d_true]` 汇总（expected lag 版本）。

- CSV: [conditional_bias_table.csv](./conditional_bias_table.csv)
