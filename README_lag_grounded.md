# Lag-Grounded DIMF 框架说明

这份文档只解释当前 lag-grounded DIMF 的框架、模块边界、核心公式和数据流。不记录具体实验结果。

## 1. 框架目标

普通多阶段时序建模容易默认同一时刻的上游和下游直接对应：

```text
stage1[t] -> stage2[t] -> stage3[t] -> y[t+H]
```

但真实过程里，上游影响下游往往有延迟。例如：

```text
stage1[t-3] -> stage2[t]
```

lag-grounded DIMF 的目标是把这个延迟关系显式建模出来：

```text
source sequence
-> lag identifier 估计 source 到 target 的 lag 分布
-> 可选 feature screening 清洗 lag prior
-> delay alignment 使用 lag prior 做跨阶段对齐
-> DIMF 融合多阶段信息并预测 y
```

核心思想是：

```text
不要只让模型隐式猜时间错位。
先显式估计 lag，再把 lag prior 用到 DIMF 的跨阶段信息传递中。
```

## 2. 基本术语

### 2.1 source 和 target

一个 lag edge 通常写成：

```yaml
lag_edges:
  - name: stage1_to_stage2
    source_stage: stage1
    target_stage: stage2
```

含义：

- `source`：上游阶段，例如 `stage1`
- `target`：下游阶段，例如 `stage2`
- `source feature`：上游阶段里的单个变量，例如 `stage1_var1`
- `target feature`：下游阶段里的单个变量，例如 `stage2_var1`

如果 edge 是：

```text
stage1_to_stage2
```

则：

```text
source = stage1
target = stage2
```

如果 edge 是：

```text
stage2_to_stage3
```

则：

```text
source = stage2
target = stage3
```

### 2.2 lag

`lag = k` 表示 target 当前时刻更可能对应 source 往前 `k` 步：

```text
target[t] 对应 source[t-k]
```

例如：

```text
lag = 0: target[t] 对应 source[t]
lag = 3: target[t] 对应 source[t-3]
lag = 5: target[t] 对应 source[t-5]
```

若 `max_lag = 5`，候选 lag 为：

```text
0, 1, 2, 3, 4, 5
```

候选数量：

```text
K = max_lag + 1
```

### 2.3 feature-level lag 和 edge-level lag

STDA lag identifier 会先对每个 source feature 预测 lag 分布：

```text
pi_lag: [B, d_source, K]
```

然后再汇总成整条 edge 的 lag 分布：

```text
pi_edge: [B, K]
```

两者区别：

```text
feature-level:
  第 i 个 source feature 自己认为 lag 是多少

edge-level:
  整个 source stage 到 target stage 的综合 lag 判断
```

## 3. 总体数据流

完整链路可以分成两个逻辑阶段。

### 3.1 阶段一：lag identifier 学 lag

这一阶段训练或加载 `STDALagIdentifier`。

```text
source_seq, target_seq
-> stda_lag_identifier.py
-> pi_lag, pi_edge, expected_lag, argmax_lag, occurrence
-> best_lag_identifier.pt
```

这一阶段类似预训练，但预训练对象是 lag 识别器，不是整个 DIMF 主模型。

它负责学习：

```text
source 到 target 延迟几步
是否存在非零 lag
哪些 source feature 对 lag 判断更重要
```

### 3.2 阶段二：DIMF 用 lag prior 做预测

这一阶段把 lag identifier 接入 DIMF。

```text
source_seq, target_seq
-> STDA 产生 feature-level lag prior
-> 可选 lag_feature_screening
-> delay_alignment
-> DIMF
-> y prediction
```

阶段二才会完整使用：

```text
stda_lag_identifier.py
lag_feature_screening.py
delay_alignment.py
dimf.py
```

## 4. 模块地图

主要代码位于 `src/`：

```text
src/
├─ data/
│  ├─ dataprocess.py
│  ├─ dataset.py
│  └─ lag_injection.py
├─ models/
│  ├─ dimf.py
│  ├─ delay_alignment.py
│  ├─ encoders.py
│  ├─ stda_lag_identifier.py
│  └─ lag_feature_screening.py
├─ metrics/
│  ├─ lag_metrics.py
│  └─ lag_visualization.py
├─ postprocess/
│  └─ viterbi_lag_decoder.py
└─ utils/
   ├─ logger.py
   ├─ metrics.py
   └─ seed.py
```

最核心的四个模型模块是：

```text
stda_lag_identifier.py
  负责预测 lag prior

lag_feature_screening.py
  负责筛选可信 source features 的 lag prior

delay_alignment.py
  负责用 lag prior 做跨阶段对齐

dimf.py
  负责整体多阶段预测框架
```

## 5. `stda_lag_identifier.py`

### 5.1 模块职责

`STDALagIdentifier` 是 lag 识别器，也可以理解为 lag prior 生产器。

它回答的问题是：

```text
给定 source stage 和 target stage 的时间窗口，
source 到 target 的 delay/lag 应该是多少？
```

它不是最终 y 预测器，也不是 delay alignment 执行器。

### 5.2 输入

主要输入：

```text
source_seq: [B, L, d_source]
target_seq: [B, L, d_target]
```

其中：

- `B`：batch size
- `L`：时间窗口长度
- `d_source`：source stage 特征数
- `d_target`：target stage 特征数
- `K = max_lag + 1`：候选 lag 数

也可以传入已经编码好的 target 表示：

```text
target_repr: [B, hidden_dim]
```

### 5.3 输出

主要输出：

```text
pi_lag:               [B, d_source, K]
pi_edge:              [B, K]
expected_lag:         [B, d_source]
expected_edge:        [B]
argmax_lag:           [B, d_source]
argmax_edge:          [B]
lag_occurrence_logit: [B, d_source]
occurrence_logit_edge:[B]
feature_importance:   [B, d_source]
scores:               [B, d_source, K]
raw_pi_lag:           [B, d_source, K]
```

含义：

- `pi_lag`：每个 source feature 的 lag 分布
- `pi_edge`：整条 edge 的综合 lag 分布
- `expected_lag`：每个 source feature 的期望 lag
- `expected_edge`：整条 edge 的期望 lag
- `argmax_lag`：每个 source feature 最可能的 lag
- `argmax_edge`：整条 edge 最可能的 lag
- `lag_occurrence_logit`：每个 source feature 是否存在非零 lag
- `feature_importance`：每个 source feature 对 edge-level lag 判断的权重

### 5.4 source 和 target 编码

target 序列通过 GRU 编码：

```text
h_target = GRU_target(target_seq)
c_t = W_target h_target
```

每个 source feature 单独通过 source GRU 得到上下文：

```text
h_i = GRU_source(source_seq[:, :, i])
```

其中 `i` 是 source feature 编号。

### 5.5 候选 lag 表示

对每个候选 lag `k`，模型需要构造 source 在 `t-k` 附近的表示。

简单模式下使用单点：

```text
source_i[t-k]
```

如果启用 `CandidateLagWindowEncoder`，则使用 causal patch：

```text
[source_i[t-k-r], ..., source_i[t-k]]
```

其中 `r = lag_window_radius`。

这个 patch 会经过 MLP 编码成：

```text
z_i,k
```

### 5.6 lag 打分

对每个 source feature `i` 和候选 lag `k`，计算匹配分数：

```text
score_i,k = <W z_i,k, c_t> / sqrt(H) + b_k
```

其中：

- `z_i,k`：第 `i` 个 source feature 在候选 lag `k` 下的表示
- `c_t`：target 上下文
- `H`：hidden dimension
- `b_k`：每个 lag 的可学习 bias

如果启用 temporal decay：

```text
score_i,k = score_i,k - gamma_i * k
```

其中：

- `gamma_i`：第 `i` 个 source feature 的可学习时间衰减强度
- `k`：lag 候选值

如果传入外部 prior：

```text
score_i,k = score_i,k + log(pi_prior_i,k)
```

### 5.7 lag 分布

先得到原始 lag 分布：

```text
raw_pi_i,k = softmax(score_i,k / tau)
```

其中：

- `tau`：temperature
- `tau` 越小，分布越尖锐
- `tau` 越大，分布越平滑

### 5.8 occurrence gate

模型还会判断是否存在非零 lag。

```text
p_i = sigmoid(MLP([h_i, c_t]))
```

其中：

- `p_i`：第 `i` 个 source feature 存在非零 lag 的概率

最终分布被拆成：

```text
pi_i,0 = 1 - p_i
pi_i,k = p_i * normalize(raw_pi_i,k), k >= 1
```

这样做的意义是：

```text
lag=0 表示 no-lag 或无明显延迟
lag>0 表示存在非零延迟
```

这个设计能把“有没有 lag”和“lag 是几”分开建模。

### 5.9 feature importance 汇总

每个 source feature 都有自己的 lag 分布。为了得到整条 edge 的 lag 分布，需要加权汇总。

先计算 feature importance：

```text
alpha_i = softmax(MLP(h_i))
```

然后：

```text
pi_edge,k = sum_i alpha_i * pi_i,k
```

edge-level expected lag：

```text
expected_edge = sum_k k * pi_edge,k
```

edge-level argmax lag：

```text
argmax_edge = argmax_k pi_edge,k
```

### 5.10 loss

`lag_identifier_loss` 主要由多项组成：

```text
L = lambda_soft_lag * L_soft
  + lambda_expected_lag * L_expected
  + lambda_occurrence * L_occurrence
  + lambda_entropy * L_entropy
  + lambda_smooth * L_smooth
  + lambda_positive_smooth * L_positive_smooth
  + lambda_positive_ce * L_positive_ce
  + lambda_shape_curvature * L_shape_curvature
```

soft lag cross entropy：

```text
L_soft = - sum_k q_k log(pi_edge,k)
```

其中 `q_k` 是 soft lag label。

expected lag loss：

```text
E_pred = sum_k k * pi_edge,k
E_gt   = sum_k k * q_k
L_expected = |E_pred - E_gt|
```

occurrence loss：

```text
L_occurrence = BCEWithLogits(occurrence_logit_edge, lag_flag)
```

positive lag CE 只在真实 `lag > 0` 的样本上计算：

```text
L_positive_ce = CE(normalize(pi_edge,k>=1), lag_gt - 1)
```

如果使用 gaussian lag label，则 positive 部分可用 KL：

```text
L_positive_kl = KL(pos_pi || gaussian(lag_gt, sigma))
```

## 6. `lag_feature_screening.py`

### 6.1 模块职责

这个模块负责筛选哪些 source feature 的 lag prior 可信。

它不是训练 STDA 必须的一步。它主要用于完整 DIMF 预测阶段：

```text
STDA 输出 feature-level pi_lag
-> feature screening 清洗不可信 feature 的 prior
-> delay alignment 使用筛过的 prior
```

### 6.2 输入输出

输入：

```text
pi_prior: [B, d_source, K]
feature_mask: [d_source]
```

输出：

```text
screened_pi_prior: [B, d_source, K]
```

### 6.3 feature score

可用分数包括：

```text
attention_mass_score
gradient_energy_score
entropy_penalty_score
ablation_importance
```

组合逻辑：

```text
score =
    normalize(attention_mass)
  + normalize(gradient_energy)
  + normalize(ablation_importance)
  - normalize(entropy_penalty)
```

其中 entropy penalty 越高，说明 lag 分布越不确定，所以会扣分。

### 6.4 选择 feature

根据 `top_k` 或 `top_ratio` 生成 mask：

```text
feature_mask[i] = True   保留第 i 个 source feature 的 prior
feature_mask[i] = False  弱化第 i 个 source feature 的 prior
```

### 6.5 弱化不可信 prior

对不可信 feature，默认替换为均匀分布：

```text
out[:, ~mask, :] = uniform
```

如果设置 `weak_prior_mix > 0`：

```text
out[:, ~mask, :] =
    weak_prior_mix * original
  + (1 - weak_prior_mix) * uniform
```

这表示：

```text
可信 feature:
  保留 STDA 给出的 lag prior

不可信 feature:
  不强行相信 STDA，退回更弱、更均匀的 prior
```

## 7. `delay_alignment.py`

### 7.1 模块职责

`DelayAlignment` 是 DIMF 里真正执行延迟对齐的模块。

它回答的问题是：

```text
下游当前时刻要融合上游信息时，
应该从上游哪个历史位置取信息？
```

它不是 lag label 训练器。它使用模型自己算出的 alignment score，或者使用 STDA 提供的 prior。

### 7.2 候选 lag

对当前时间 `t`，构造候选：

```text
lag = 0 -> source[t]
lag = 1 -> source[t-1]
lag = 2 -> source[t-2]
...
lag = L_max -> source[t-L_max]
```

候选上游表示：

```text
up_raw_k = source[t-k]
```

### 7.3 alignment score

用下游当前状态作为 query，上游候选 lag 表示作为 key：

```text
alpha_k = <Wq down_t, Wk up_raw_k> / sqrt(A)
```

其中：

- `down_t`：下游当前表示
- `up_raw_k`：上游在 lag `k` 下的候选表示
- `A`：attention dimension

如果启用 lag embedding：

```text
key_input_k = up_raw_k + emb(k)
```

如果启用 lag bias：

```text
alpha_k = alpha_k + b_k
```

### 7.4 使用 delay prior

如果外部提供 `pi_prior`：

```text
alpha_k = alpha_k + lambda_prior * log(pi_prior_k)
```

如果外部提供 expected lag `d_prior`，可以构造 gaussian prior：

```text
q_k = exp(-0.5 * ((k - d_prior) / sigma_prior)^2)
q = normalize(q)
```

再注入：

```text
alpha_k = alpha_k + lambda_prior * log(q_k)
```

支持的 prior mode 包括：

```text
none
soft_distribution
gaussian_from_expected
```

### 7.5 lag 权重和消息融合

得到 lag 分布：

```text
pi_k = softmax(alpha_k / tau)
```

融合上游信息：

```text
message = sum_k pi_k * Wv(up_raw_k)
```

保留未投影版本：

```text
raw_message = sum_k pi_k * up_raw_k
```

输出：

```text
message
pi
raw_message
```

### 7.6 `NoDelayAlignment`

`NoDelayAlignment` 是对照模块。

它不做显式 delay search，而是退化为：

```text
lag = 0
```

即默认同一时刻对齐。

## 8. `dimf.py`

### 8.1 模块职责

`DIMF` 是主模型。它负责：

```text
编码各阶段序列
跨阶段融合信息
调用 delay alignment
接收或内部生成 lag prior
输出最终 y prediction
```

`dimf.py` 本身不是单独的 lag 识别器。它可以挂载一个 lag identifier，然后在 forward 时生成 delay prior。

### 8.2 attach lag identifier

DIMF 支持把训练好的 STDA lag identifier 挂上去：

```text
dimf.attach_lag_identifier(...)
```

挂载后，DIMF 可以在内部调用：

```text
source_seq, target_seq
-> lag_identifier
-> pi_lag
-> optional feature mask
-> delay_prior payload
```

### 8.3 内部 prior payload

给 delay alignment 的 prior 通常包含：

```text
pi_prior
d_prior
prior_mode
lambda_prior
sigma_prior
```

最常用的是：

```text
pi_prior: [B, d_source, K] 或 [B, K]
prior_mode: soft_distribution
```

### 8.4 跨阶段边

DIMF 可对不同阶段边分别开启或关闭 alignment：

```text
feed_to_stage1
stage1_to_stage2
stage2_to_stage3
```

如果某条边开启 alignment，则使用 `DelayAlignment`。

如果关闭，则使用 `NoDelayAlignment`。

## 9. `lag_injection.py`

### 9.1 模块职责

`src/data/lag_injection.py` 用于构造带 lag 标签的人工数据。

它可以在指定 edge 上注入：

```text
source -> target
```

例如：

```text
stage1 -> stage2
```

### 9.2 注入逻辑

对于每个被选中的时间点 `t`，先根据 lag 分布 `q` 构造延迟混合：

```text
lagged(t) = sum_k q_k * source[t-k]
```

再写入 target stage：

```text
target_injected[t] =
    (1 - rho) * target_original[t]
  + rho * scaled(lagged(t))
```

其中：

- `q_k`：lag soft label
- `rho`：注入强度
- `scaled`：把 source 的标准化值映射到 target column 的尺度

### 9.3 写入标签

注入后会写入 lag 监督字段，例如：

```text
lag_gt
lag_expected_gt
lag_flag
shape_id
shape_type
{edge}_true_pi_lag0 ... {edge}_true_pi_lagK
```

这些字段主要用于训练和评估 STDA lag identifier。

### 9.4 y target 是否应该移动

`stage1 -> stage2` 的 lag 注入表示：

```text
stage2[t] 部分来自 stage1[t-k]
```

它不自动表示：

```text
y[t] 应该平移成 y[t-k]
```

是否修改最终 `y` 是数据构造策略问题：

```text
只训练 lag identifier:
  通常不需要修改 y。

完整 y prediction 且希望 y 明确响应人工 lag:
  可以额外设计 y response 机制。
```

更合理的 y response 应该来自被修改后的过程变量，而不是简单平移 y：

```text
y_injected[t] = y_original[t] + beta * f(stage2_injected[t] - stage2_original[t])
```

如果使用人为一一映射，也应该把它视为合成假设，例如：

```text
lag k -> y 增加 k%
```

这类规则适合机制验证，但不是默认物理规律。

## 10. 后处理和指标模块

### 10.1 `postprocess/viterbi_lag_decoder.py`

用于把逐点 lag 分布解码成更平滑的 lag 路径。

直觉是：

```text
逐点 argmax 可能抖动
Viterbi 加入切换惩罚和平滑约束
得到更连续的 lag 序列
```

### 10.2 segment-level 稳定化

一些流程会在评估阶段做 segment-level 后处理，例如：

```text
segment zero gate
segment mode filter
```

作用是减少 no-lag 区域误报和局部抖动。

注意：

```text
后处理能提升最终 lag recovery 稳定性，
但也会遮住 raw 模型输出的随机性和差异。
```

因此分析模型本身时，应该区分：

```text
raw prediction
postprocessed prediction
stable prediction
```

### 10.3 `metrics/lag_metrics.py`

用于计算 lag 识别指标，例如：

```text
expected_lag_mae
argmax_lag_accuracy
soft_kl
soft_js
occurrence_auprc
no_lag_false_alarm_rate
```

### 10.4 `metrics/lag_visualization.py`

用于生成 lag 分布、expected lag 曲线、segment block 等可视化。

## 11. 模块边界总结

最容易混淆的是这三者：

```text
stda_lag_identifier.py
  预测 lag 是多少
  输出 pi_lag / pi_edge / expected_lag / occurrence

lag_feature_screening.py
  判断哪些 source feature 的 lag prior 可信
  对不可信 feature 的 prior 做弱化

delay_alignment.py
  在 DIMF 内部真正使用 lag prior
  把 source[t-k] 按 lag 分布融合到 target 当前表示
```

一句话总结：

```text
STDA 负责找 lag。
feature screening 负责筛可信 prior。
delay alignment 负责用 lag 对齐。
DIMF 负责最终预测。
```

## 12. 典型调用关系

### 12.1 只训练 lag identifier

```text
dataset
-> source_seq, target_seq
-> STDALagIdentifier
-> lag_identifier_loss
-> best_lag_identifier.pt
```

这个路径主要验证：

```text
lag 识别器本身准不准
```

不需要使用：

```text
lag_feature_screening.py
delay_alignment.py
```

### 12.2 完整 DIMF 使用 lag prior

```text
dataset
-> DIMF
   -> internal lag identifier
      -> pi_lag
      -> optional feature screening
      -> pi_prior
   -> delay alignment
      -> aligned message
   -> y prediction
```

这个路径验证：

```text
显式 lag prior 是否能帮助最终 y 预测
```

### 12.3 没有 lag prior 的 DIMF

```text
dataset
-> DIMF
-> DelayAlignment 自己学习 alignment
-> y prediction
```

或：

```text
dataset
-> DIMF
-> NoDelayAlignment
-> y prediction
```

这类路径用于对照：

```text
没有外部 lag prior 时 DIMF 表现如何
```

## 13. 配置中的关键字段

### 13.1 lag identifier

```yaml
lag_identifier:
  enabled: true
  model: stda_lag_identifier
  max_lag: 5
  hidden_dim: 64
  use_candidate_window_encoder: true
  lag_window_radius: 2
  use_feature_attention: true
  use_sequence_smoother: true
```

关键含义：

- `max_lag`：最大候选延迟
- `use_candidate_window_encoder`：是否用局部 causal patch 表示候选 lag
- `lag_window_radius`：patch 往前看的半径
- `use_feature_attention`：是否用 feature importance 汇总 feature-level lag
- `use_sequence_smoother`：是否对连续样本的 lag 输出做序列平滑

### 13.2 delay prior

```yaml
delay_prior:
  enabled: true
  prior_mode: soft_distribution
  lambda_prior: 1.0
  sigma_prior: 1.5
  weak_prior_mix: 0.0
```

关键含义：

- `enabled`：是否启用外部 delay prior
- `prior_mode`：如何把 prior 注入 alignment
- `lambda_prior`：prior 强度
- `sigma_prior`：用 expected lag 构造 gaussian prior 时的宽度
- `weak_prior_mix`：不可信 feature 的 prior 和均匀分布混合比例

### 13.3 feature screening

```yaml
feature_screening:
  enabled: true
  top_ratio: 1.0
  top_k: null
```

关键含义：

- `enabled`：是否启用 feature screening
- `top_ratio`：保留多少比例的 source features
- `top_k`：直接指定保留几个 source features

如果 `top_ratio = 1.0`，表示 screening 模块会执行，但不会剔除 feature。

### 13.4 model alignment

```yaml
model:
  use_alignment: true
  align_tau: 0.7
  lag_emb: true
  use_lag_bias: true
  lag_head_mode: softmax
```

关键含义：

- `use_alignment`：是否使用 delay alignment
- `align_tau`：alignment softmax temperature
- `lag_emb`：是否给 lag 候选加可学习 embedding
- `use_lag_bias`：是否给每个 lag 加可学习 bias
- `lag_head_mode`：lag 分布头形式

## 14. 快速判断一个模块是否参与了当前流程

看是否只训练 lag identifier：

```text
只有 lag 预测、lag metrics、lag checkpoint
-> 通常只用了 stda_lag_identifier.py
```

看是否进入完整 DIMF y prediction：

```text
有 DIMF checkpoint、prediction metrics、y prediction output
-> 通常进入了 dimf.py 主预测链路
```

看是否使用 feature screening：

```text
有 feature_mask 或 feature screening report
-> lag_feature_screening.py 至少执行过
```

看是否使用 delay alignment：

```text
DIMF 配置 use_alignment=true
且模型经过 stage edge 融合
-> delay_alignment.py 参与跨阶段对齐
```

看是否使用外部 STDA prior：

```text
delay_prior.enabled=true
并且 DIMF 挂载或加载了 lag identifier
-> STDA prior 参与 delay alignment
```
