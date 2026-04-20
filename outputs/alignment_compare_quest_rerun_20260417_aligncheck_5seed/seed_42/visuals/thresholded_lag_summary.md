# Thresholded Lag Summary

展示规则统一定义为：

```text
if P(lag>0) < tau: predicted lag = 0
else: predicted lag = argmax over non-zero lag bins
```

其中 `P(lag>0) = 1 - pi_lag0`。

## Metrics

| tau | model | pred_nonzero_rate | precision | recall | F1 | overall exact | lagged-only exact | overall MAE | lagged-only MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | aligned | 0.292 | 0.014 | 0.229 | 0.027 | 0.693 | 0.000 | 0.515 | 2.458 |
| 0.30 | noalign | 0.000 | 0.000 | 0.000 | 0.000 | 0.982 | 0.000 | 0.049 | 2.667 |
| 0.50 | aligned | 0.113 | 0.027 | 0.167 | 0.047 | 0.871 | 0.000 | 0.230 | 2.500 |
| 0.50 | noalign | 0.000 | 0.000 | 0.000 | 0.000 | 0.982 | 0.000 | 0.049 | 2.667 |

## Figure Files

### tau = 0.30

- Thresholded block plot: `lag_block_panels_threshold_pgt0_0p3.png`
- Thresholded block summary: `lag_block_summary_threshold_pgt0_0p3.csv`

### tau = 0.50

- Thresholded block plot: `lag_block_panels_threshold_pgt0_0p5.png`
- Thresholded block summary: `lag_block_summary_threshold_pgt0_0p5.csv`

