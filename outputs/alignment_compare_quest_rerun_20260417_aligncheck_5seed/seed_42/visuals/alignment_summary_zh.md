# Raw-Gap Alignment 结果汇总

## 结论

1. 预测 `yield_flow` 时，alignment 打开后 `MAE` 从 `5.757` 升到 `6.230`，这一轮预测指标没有收益。
2. 只看真正加了 lag 的样本，alignment 的 `expected lag MAE` 从 `2.667` 降到 `1.773`，说明它确实在恢复 lag 大小。
3. `argmax lag accuracy` 仍然是 `0`；在 lagged-only 样本上，aligned 的 `argmax lag` 均值为 `0.000`（众数 `0`），noalign 为 `0.000`（众数 `0`）；aligned 的 `expected lag` 均值为 `0.965`。

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
| Forecast MAE | 6.230 | 5.757 |
| Forecast RMSE | 8.800 | 8.284 |
| Forecast R2 | 0.337 | 0.413 |
| Lagged-only expected lag MAE | 1.773 | 2.667 |
| Lagged-only argmax acc | 0.000 | 0.000 |
| Mean predicted lag on lagged samples | 0.965 | 0.000 |

## 分真实 lag

| true lag | n | aligned MAE | noalign MAE | aligned pred mean | noalign pred mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 2564 | 1.062 | 0.000 | 1.062 | 0.000 |
| 1 | 12 | 0.424 | 1.000 | 0.801 | 0.000 |
| 2 | 16 | 1.050 | 2.000 | 0.994 | 0.000 |
| 3 | 8 | 1.960 | 3.000 | 1.040 | 0.000 |
| 4 | 4 | 3.053 | 4.000 | 0.947 | 0.000 |
| 5 | 4 | 3.944 | 5.000 | 1.056 | 0.000 |
| 6 | 4 | 4.884 | 6.000 | 1.116 | 0.000 |

## Lag Block 摘要

| block | start | end | true lag | aligned pred mean | noalign pred mean | aligned MAE | noalign MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-03-25 07:15 | 2022-03-25 09:00 | 1 | 1.287 | 0.000 | 0.434 | 1.500 |
| 2 | 2022-04-07 12:45 | 2022-04-07 14:30 | 2 | 2.033 | 0.000 | 2.053 | 4.000 |
| 3 | 2022-04-13 01:45 | 2022-04-13 03:30 | 2 | 0.253 | 0.000 | 3.747 | 4.000 |
| 4 | 2022-04-18 17:30 | 2022-04-18 19:15 | 1 | 1.014 | 0.000 | 1.596 | 2.500 |
| 5 | 2022-04-27 16:45 | 2022-04-27 18:30 | 1 | 0.935 | 0.000 | 1.571 | 2.500 |
| 6 | 2022-04-29 10:15 | 2022-04-29 12:00 | 1 | 0.266 | 0.000 | 1.234 | 1.500 |

## Conditional Bias Table

按 true lag 分组的 `E[\hat d | d_true]` 汇总（expected lag 版本）。

- CSV: [conditional_bias_table.csv](./conditional_bias_table.csv)
