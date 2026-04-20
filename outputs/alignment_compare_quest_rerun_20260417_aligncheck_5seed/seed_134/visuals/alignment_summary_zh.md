# Raw-Gap Alignment 结果汇总

## 结论

1. 预测 `yield_flow` 时，alignment 打开后 `MAE` 从 `6.449` 降到 `6.388`，有小幅收益。
2. 只看真正加了 lag 的样本，alignment 的 `expected lag MAE` 从 `2.667` 降到 `1.657`，说明它确实在恢复 lag 大小。
3. `argmax lag accuracy` 仍然是 `0`；在 lagged-only 样本上，aligned 的 `argmax lag` 均值为 `0.042`（众数 `0`），noalign 为 `0.000`（众数 `0`）；aligned 的 `expected lag` 均值为 `1.233`。

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
| Forecast MAE | 6.388 | 6.449 |
| Forecast RMSE | 9.047 | 9.076 |
| Forecast R2 | 0.299 | 0.295 |
| Lagged-only expected lag MAE | 1.657 | 2.667 |
| Lagged-only argmax acc | 0.021 | 0.000 |
| Mean predicted lag on lagged samples | 1.233 | 0.000 |

## 分真实 lag

| true lag | n | aligned MAE | noalign MAE | aligned pred mean | noalign pred mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 1.145 | 0.000 | 1.145 | 0.000 |
| 1 | 12 | 0.470 | 1.000 | 1.145 | 0.000 |
| 2 | 16 | 0.992 | 2.000 | 1.216 | 0.000 |
| 3 | 8 | 1.625 | 3.000 | 1.375 | 0.000 |
| 4 | 4 | 2.374 | 4.000 | 1.626 | 0.000 |
| 5 | 4 | 3.943 | 5.000 | 1.057 | 0.000 |
| 6 | 4 | 4.943 | 6.000 | 1.057 | 0.000 |

## Lag Block 摘要

| block | start | end | true lag | aligned pred mean | noalign pred mean | aligned MAE | noalign MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-03-25 07:15 | 2022-03-25 09:00 | 1 | 0.848 | 0.000 | 0.652 | 1.500 |
| 2 | 2022-04-07 12:45 | 2022-04-07 14:30 | 2 | 1.819 | 0.000 | 2.181 | 4.000 |
| 3 | 2022-04-13 01:45 | 2022-04-13 03:30 | 2 | 0.271 | 0.000 | 3.729 | 4.000 |
| 4 | 2022-04-18 17:30 | 2022-04-18 19:15 | 1 | 0.802 | 0.000 | 1.698 | 2.500 |
| 5 | 2022-04-27 16:45 | 2022-04-27 18:30 | 1 | 2.613 | 0.000 | 1.035 | 2.500 |
| 6 | 2022-04-29 10:15 | 2022-04-29 12:00 | 1 | 1.044 | 0.000 | 0.648 | 1.500 |

## Conditional Bias Table

按 true lag 分组的 `E[\hat d | d_true]` 汇总（expected lag 版本）。

- CSV: [conditional_bias_table.csv](./conditional_bias_table.csv)
