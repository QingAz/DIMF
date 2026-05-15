# Lag-Mapped Y Target Dataset

This dataset keeps the existing stage1-to-stage2 lag injection and adds a one-to-one y response.

Rule:

```text
yield_flow = yield_flow_original * (1 + lag_gt * 0.01)
```

With the default mapping, lag 0 is unchanged, lag 3 maps to +3%, and lag 5 maps to +5%.
