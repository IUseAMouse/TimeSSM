# TimeSSM-2.5M

<!-- DRAFT 2026-10-08. One checkpoint only, the released one: bare and with the full inference stack; no band of checkpoints, no 10M line (user decisions 2026-10-08). Cells marked [ ] are filled from the evaluation logs; every number
     in this card must be traceable to a per_config/ directory on the pod or to an official
     leaderboard file. Nothing here is rounded beyond what the harness prints. -->

A 2.5M-parameter zero-shot forecaster built on a diagonal linear state space model (S4D),
trained from scratch on a 16-billion-observation corpus of public time series, on three
RTX 3090s. It forecasts any horizon in one pass, outputs nine quantiles, and exposes its
sampling interval as an inference-time knob.

Evaluated on GIFT-Eval (97 configurations) with an evaluation harness checked against the
official leaderboard files: it reproduces t0-beta's published per-configuration results on
97 of 97 configurations.

## Results

GIFT-Eval, 97 configurations, zero-shot. MASE and CRPS are geometric means of the ratio to
the official seasonal naive, as on the leaderboard. Lower is better.

### The released checkpoint, layer by layer

Checkpoint `epoch00_valloss1.2814` [numbers below still those of 1.2841: to update]. Each row adds one inference-time layer to the previous
one; the model's weights never change.

| Inference setting | MASE | CRPS | 80% coverage |
|---|---|---|---|
| Bare model | 0.8534 | 0.5836 | [ ] |
| + sign-flip averaging | 0.8343 | 0.5644 | [ ] |
| + backtest-selected resampling (mix, pooled) | 0.7671 | 0.5242 | [ ] |
| + upsampling candidates, four backtest windows | 0.7613 | 0.5194 | 0.699 |
| + sampling interval set from the declared frequency | 0.7572 | 0.5155 | 0.696 |
| + quantile widening calibrated on the training corpus | **0.7572** | **0.5142** | **0.758** |

The last row is the card's number. One command reproduces it (see Reproduction).

### Where it stands

| Model | Parameters | MASE | CRPS | Source |
|---|---|---|---|---|
| FlowState-9.1M | 9.1M | 0.7262 | 0.5019 | leaderboard |
| **TimeSSM-2.5M** (this card) | 2.5M | 0.7572 | 0.5142 | this harness |
| TTM-R3-PT | 1.4M | 0.7240 | 0.5195 | leaderboard (saw GIFT-Eval in pretraining) |
| Toto-2.0-4m | 4M | 0.7565 | 0.5242 | leaderboard |
| Chronos-Bolt small, bare | 48M | 0.8259 | 0.5636 | this harness, chronos-forecasting 2.3.2 |
| Chronos-Bolt small + the same resampling layer | 48M | 0.7467 | 0.5148 | this harness |

### By frequency and by term

[Table from gift_gap_ssm.py on the published setting: short / medium / long, and the ten
frequencies, MASE and CRPS, with the number of configurations in each group.]

## What the numbers rely on

Read this before quoting the card.

- **Frequency rule.** The sampling interval is set to 24 / season, where the season is the
  reference cycle of the declared sampling frequency. This is FlowState's convention,
  including two pieces of GIFT-Eval knowledge taken from its reference implementation: the
  dataset's domain decides whether daily data has a weekly cycle (transport, healthcare,
  sales), and `bizitobs_l2c` is treated as having no daily cycle. The model therefore
  receives the frequency, like FlowState does. Without this rule the number is 0.7613 /
  0.5194.
- **Quantile widening.** Nine per-level factors applied to the fan around the median,
  calibrated on validation windows of the training corpus under the same inference setting.
  GIFT-Eval data were never used to fit them. The median, hence MASE, is untouched.
- **Backtest-selected resampling.** For each configuration the context is resampled at
  several rates (decimation by 2 to 48, upsampling by 2 to 4); a causal backtest on the
  series' own past, before the first test target, weights the candidates; the forecasts
  are mixed per quantile level. It costs 20 to 40 forecasts per series instead of one.
  `ratein` is the internal name of this layer in the code.
- **Zero-shot.** The training corpus excludes the GIFT-Eval sources by construction; it
  includes synthetic series and decimated copies of some real ones.
- **Checkpoint.** The released checkpoint is the last one of its training run; no
  checkpoint was picked on GIFT-Eval.
- **Harness fidelity.** The harness recomputes the leaderboard metrics from the raw data.
  t0-beta's official per-configuration results are reproduced on 97/97 configurations;
  Chronos-Bolt's differ on medium and long horizons (its autoregressive rollout changed
  between library versions), so Chronos-Bolt numbers above are those of version 2.3.2.

## Model

- Architecture: [n] blocks of gated S4D (complex diagonal state, d_model [ ], d_state 32),
  one token per time step, a learned future token for the horizon, a quantile head with
  cross-attention on the context. [Exact counts from `get_num_params`.]
- Input normalization: robust arcsinh scaling then instance normalization, both from the
  context.
- Horizon: any length, no autoregressive loop; GIFT-Eval asks for 6 to 900 steps.
- Rate knob: every layer's step size is multiplied by `w`; `w = 1/k` is equivalent to
  decimating the input by `k` (tested).

## Training

- Corpus: 106 files, 15.85 billion observations: LOTSA subsets (revision pinned), four
  Chronos datasets, synthetic families, decimated copies. [Link the manifest and the
  frequency table.]
- Objective: pinball loss on nine quantiles, in the normalized frame.
- Three stages on 3x RTX 3090: [spike run: steps, windows]; wide-range sampling-interval
  augmentation (continuation, 89M windows); [if the frequency-tied arm is adopted: its
  line]. Optimizer AdamW, cosine schedule. [Exact budgets from the registry.]
- Validation: 2% of each file, used for the calibration factors and for checkpointing;
  never GIFT-Eval.

## Reproduction

```bash
# TimeJEPA (harness, data loaders) and TimeMamba (model) side by side; GIFT-Eval data via `make gift-download`
EVAL_CONFIG=timessm_2.5m_gift scripts/eval_ssm.sh <path/to/epoch00_valloss1.2841.ckpt>
# bare model, no inference layer:
scripts/eval_ssm.sh <path/to/epoch00_valloss1.2841.ckpt>
```

The first command carries every setting of the card's number (configs/timessm_2.5m_gift.yaml)
and the calibration file (configs/calibration/). Expected: MASE 0.7572, CRPS 0.5142.
[Confirmed on a fresh directory on YYYY-MM-DD.]

## Limitations

- The bare model is a short-horizon model: its median is worse than the seasonal naive
  beyond about 480 steps, and the resampling layer is what carries the medium and long
  terms. [Numbers from the nu/stack map.]
- Weakest groups: hourly data with a weekly cycle (m4_hourly is worse than the seasonal
  naive even with every layer), low-frequency economic series (m4 weekly and yearly:
  level errors on trending series), cloud-operations traces at 10-second resolution.
- The resampling layer multiplies inference cost by 20 to 40.
- Univariate: multivariate datasets are forecast one variable at a time.

## Files, license, citation

[Weights (checkpoint, state dict), config, calibration file, frequency table, manifest.
License. Git tag of the evaluated code in both repositories. How to cite.]
