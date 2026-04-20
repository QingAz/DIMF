# Raw-Gap Alignment 结果汇总

## 结论

1. 预测 `yield_flow` 时，alignment 打开后 `MAE` 从 `6.152` 升到 `7.303`，这一轮预测指标没有收益。
2. 只看真正加了 lag 的样本，alignment 的 `expected lag MAE` 从 `2.667` 降到 `2.434`，说明它确实在恢复 lag 大小。
3. `argmax lag accuracy` 仍然是 `0`；在 lagged-only 样本上，aligned 的 `argmax lag` 均值为 `0.000`（众数 `0`），noalign 为 `0.000`（众数 `0`）；aligned 的 `expected lag` 均值为 `0.232`。

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
| Forecast MAE | 7.303 | 6.152 |
| Forecast RMSE | 9.941 | 8.654 |
| Forecast R2 | 0.154 | 0.359 |
| Lagged-only expected lag MAE | 2.434 | 2.667 |
| Lagged-only argmax acc | 0.000 | 0.000 |
| Mean predicted lag on lagged samples | 0.232 | 0.000 |

## 分真实 lag

| true lag | n | aligned MAE | noalign MAE | aligned pred mean | noalign pred mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 0.242 | 0.000 | 0.242 | 0.000 |
| 1 | 12 | 0.851 | 1.000 | 0.149 | 0.000 |
| 2 | 16 | 1.784 | 2.000 | 0.216 | 0.000 |
| 3 | 8 | 2.719 | 3.000 | 0.281 | 0.000 |
| 4 | 4 | 3.767 | 4.000 | 0.233 | 0.000 |
| 5 | 4 | 4.647 | 5.000 | 0.353 | 0.000 |
| 6 | 4 | 5.671 | 6.000 | 0.329 | 0.000 |

## Lag Block 摘要

| block | start | end | true lag | aligned pred mean | noalign pred mean | aligned MAE | noalign MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-03-25 07:15 | 2022-03-25 09:00 | 1 | 0.135 | 0.000 | 1.365 | 1.500 |
| 2 | 2022-04-07 12:45 | 2022-04-07 14:30 | 2 | 0.588 | 0.000 | 3.412 | 4.000 |
| 3 | 2022-04-13 01:45 | 2022-04-13 03:30 | 2 | 0.105 | 0.000 | 3.895 | 4.000 |
| 4 | 2022-04-18 17:30 | 2022-04-18 19:15 | 1 | 0.076 | 0.000 | 2.424 | 2.500 |
| 5 | 2022-04-27 16:45 | 2022-04-27 18:30 | 1 | 0.400 | 0.000 | 2.100 | 2.500 |
| 6 | 2022-04-29 10:15 | 2022-04-29 12:00 | 1 | 0.090 | 0.000 | 1.410 | 1.500 |

## Conditional Bias Table

按 true lag 分组的 `E[\hat d | d_true]` 汇总（expected lag 版本）。

- CSV: [conditional_bias_table.csv](./conditional_bias_table.csv)
