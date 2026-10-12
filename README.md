# TimeSSM

![TimeSSM-2.5M forecasting a fluid simulation it has never seen, every pixel as an independent series: ground truth, forecast median, forecast mean, uncertainty](docs/assets/timessm_letters.gif)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

A 2.5M-parameter zero-shot forecaster built on a diagonal linear
time-invariant state space model (S4D) whose sampling interval is an
inference-time parameter: one checkpoint runs a series at any rate. Trained
from scratch on 15.85 billion observations of public time series, on three
RTX 3090s, by one person. Branch `timessm`. The March 2025 Mamba code this
repository started from is kept under [legacy/](legacy/) with its own
`pyproject`, untouched.

## Results

GIFT-Eval, 97 configurations, zero-shot, geometric means of the ratio to the
official seasonal naive. One checkpoint, the last of its training run.

| Inference | MASE ratio | CRPS ratio |
|---|---|---|
| TimeSSM-2.5M + test-time backtest resampling | **0.7544** | **0.5128** |
| TimeSSM-2.5M | 0.7985 | 0.5451 |

The first row resamples each context at several rates, weights the candidates
by a causal backtest on the series' own past, mixes the forecasts per quantile
level and widens the fan with factors calibrated on the training corpus;
nothing is fitted on GIFT-Eval. The second row is one forward pass. The
weights are on Hugging Face at
[IUseAMouse/timessm-2.5m](https://huggingface.co/IUseAMouse/timessm-2.5m),
with the model card and the 97 per-configuration numbers (a copy of the card
is in [docs/model_card_timessm_2.5m.md](docs/model_card_timessm_2.5m.md)).

Among the models of the GIFT-Eval leaderboard under 10M parameters, sorted by
CRPS (leaderboard numbers and parameter counts from its metadata; the two
TimeSSM rows and the Chronos-Bolt rows from this harness):

| Model | Parameters | CRPS ratio |
|---|---|---|
| FlowState-9.1M | 9.1M | 0.502 |
| **TimeSSM-2.5M + test-time backtest resampling** | **2.5M** | **0.513** |
| TTM-R3 | 1.4M | 0.520 |
| Toto-2.0-4m | 4.1M | 0.524 |
| Chronos-Bolt tiny + the same resampling layer | 9M | 0.532 |
| TempoPFN | n/a | 0.533 |
| TinyCast | 0.1M | 0.545 |
| **TimeSSM-2.5M** | **2.5M** | **0.545** |
| goia-forecast-nano | 4.7M | 0.553 |
| Kairos-10m | 9.9M | 0.554 |
| Metamorph-4.5M | n/a | 0.555 |
| Chronos-Bolt tiny | 9M | 0.601 |
| YingLong-6m | 7.3M | 0.609 |

TTM-R3 saw GIFT-Eval data in pretraining. Above 10M for scale: Chronos-Bolt
small (48M) is at 0.564 bare and 0.515 with the same resampling layer.

## RateIN: test-time resampling by backtest

The project started from an observation on the ETT datasets, where the gap
between foundation models and simple baselines was largest: models seemed to
have a preferred sampling regime, and presenting a series at that regime,
by decimating or interpolating its context, improved their forecasts far more
than any change to the model did. Two things followed. The model became a
state space model, because a continuous-time linear time-invariant system
discretized with a zero-order hold is equivariant to the sampling rate:
running a series decimated by k at step Delta is the same computation as
running the full series at Delta / k. FlowState ([Graf et al., 2025](https://arxiv.org/abs/2508.05287))
built a forecaster on this property and ties the step to the declared
sampling frequency; TimeSSM does the same, and its layer is S4D rather than
Mamba because an input-dependent step breaks the equivariance. And the
inference layer became RateIN: the context is resampled at several rates
(decimation by 2 to 48, upsampling by 2 to 4), forecast at each, a causal
backtest on the series' own past weights the candidates, and the forecasts
are mixed per quantile level, with sign-flip averaging. Nothing is fitted on
the test data.

RateIN is model-agnostic. The table runs it on five third-party models
through the same harness, and adds the oracle, the best single rate per
configuration chosen on the test set: a diagnostic ceiling, never an official
number. GIFT-Eval, 97 configurations, CRPS ratio to the seasonal naive
(MASE for TTM-R3, which is a point forecaster).

| Model | Parameters | Bare | + RateIN | Oracle rate | Gain | Share of the ceiling |
|---|---|---|---|---|---|---|
| TimeSSM-2.5M | 2.5M | 0.5451 | **0.5128** | 0.5069 | -5.9 % | 85 % |
| Chronos-Bolt tiny | 9M | 0.6010 | 0.5315 | 0.5192 | -11.6 % | 84 % |
| Chronos-Bolt small | 48M | 0.5636 | 0.5148 | 0.5010 | -8.7 % | 78 % |
| TTM-R3 (MASE) | 1.4M | 0.7635 | 0.7475 | 0.7324 | -2.1 % | 33 % |
| Chronos-2, univariate | 120M | 0.4822 | 0.4763 | 0.4667 | -1.2 % | 19 % |
| t0-beta | 256M | 0.4737 | 0.4651 | 0.4561 | -1.8 % | 46 % |

Two readings. The ceiling shrinks with the strength of the model, from 13 %
on Bolt tiny to 2.5 % on Chronos-2: strong models trained on many
frequencies have less left to take. And the share of the ceiling that the
backtest captures shrinks too, from 85 % to 19 %: on strong models the gains
per configuration are a few percent and the backtest no longer separates
them from noise. The configurations that gain are the same for every model
(10-second and 5-minute operations traces, 10-minute solar, 5-minute
traffic, 15-minute ETT). The share of the ceiling is computed from the
sign-flipped forecast, which the oracle includes.

The cost is the limit: RateIN runs 20 to 40 forecasts per series instead of
one, 3 to 12 hours for a full GIFT-Eval run on a single GPU against 15 to 35
minutes bare. Moving the whole selection onto the model's rate knob, so that
one pass serves every rate, is what I am working on now.

## How to use

```bash
pip install git+https://github.com/IUseAMouse/TimeSSM.git
```

```python
import numpy as np
import torch
from timessm import from_pretrained, delta_scale

model = from_pretrained("IUseAMouse/timessm-2.5m")   # or a local directory; device="cuda" to run there

series = np.sin(np.arange(1024) * 2 * np.pi / 24)   # your history: raw values, any scale
context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)   # [1, length]
w = delta_scale(24, *model.delta_range)   # season length in steps: 24 for hourly data with a daily cycle
with torch.no_grad():
    out = model.forecast(context, 48, w=w)
quantiles = out["quantiles_denorm"][0]   # [48, 9], levels 0.1 to 0.9; column 4 is the median
```

The first argument of `delta_scale` is the season length of the declared
frequency, in steps: 96 for 15-minute data, 24 for hourly, 7 for daily data
with a weekly cycle, 12 for monthly. The function divides 24 by it and clamps
the result to the trained range, so `w` is 0.25 for 15-minute data and 1 for
hourly data. Normalization lives inside the model: feed raw values, read raw
values back. The test-time resampling layer runs through the evaluation
harness (`scripts/eval_ssm.sh`, below).

## Layout

```
src/timessm/ssm.py        S4DLayer: ZOH discretization, FFT convolution, step recurrence, delta_scale
src/timessm/block.py      GatedSSMBlock (norm, gate, S4D, optional readout gate; no conv by default)
src/timessm/model.py      SSMForecaster: RobustScale + RevIN + per-step tokens + blocks + quantile head,
                          autonomous rollout through a learned future token, forecast(x, n, w)
src/timessm/training.py   SSMFinetuneModule: multi-rate training on Delta, frequency-tied Delta
scripts/train_ssm.py      Hydra entry point (datamodule, checkpointing, W&B)
scripts/eval_ssm.sh       GIFT-Eval through the harness (+ratein=delta is the knob)
configs/ssm_mini_v3.yaml  the from-scratch recipe; ssm_mini_v3_wide, ssm_mini_v3_freq: the two continuations
configs/timessm_2.5m_gift.yaml  every inference setting of the released number, one command
tests/                    layer (rate equivariance first), model contract, harness
docs/GIFT_RANKINGS.md     GIFT-Eval foundation-model rankings from a vendored leaderboard snapshot
docs/RUNBOOK.md           setup, train, eval commands
```

## Setup

Inference needs only the package and torch:

```bash
pip install git+https://github.com/IUseAMouse/TimeSSM.git
```

Training and the GIFT-Eval harness use the `timejepa` package, behind the
`train` extra (a checkout next to this repo, `../TimeJEPA`, is used when
present through `[tool.uv.sources]`, the GitHub URL otherwise):

```bash
uv sync --extra train --extra dev
uv run pytest -q           # about 5 minutes on CPU
```

## Train and evaluate

```bash
# the released number (backtest resampling, calibrated fan, frequency rule)
EVAL_CONFIG=timessm_2.5m_gift scripts/eval_ssm.sh checkpoints/timessm_mini_v3_freq_zs/pretrain_False/epoch00_valloss1.2814.ckpt
# bare model, frequency given
EVAL_CONFIG=ssm_mini_v3_freq_eval scripts/eval_ssm.sh <ckpt> +freq_delta=true
# training, three stages: from scratch, wide-range rate augmentation, frequency-tied rate
python scripts/train_ssm.py --config-name ssm_mini_v3 wandb.run_name=ssm-mini-v3
scripts/train_ssm_loop.sh ssm_mini_v3_wide ssm-mini-v3-wide '+training.pretrained_encoder_path=<spike ckpt>'
scripts/train_ssm_loop.sh ssm_mini_v3_freq ssm-mini-v3-freq '+training.pretrained_encoder_path=<wide ckpt>'
```

The frequency rule sets the model's sampling interval to 24 / season of the
declared frequency (FlowState's convention), at training and at inference; the
per-file table is `../TimeJEPA/configs/corpus_v3_frequencies.yaml`.

## Rate equivariance, tested

`tests/test_ssm_layer.py` pins the property before anything else: on a
block-constant input (blocks of k steps) the layer at Delta on the fine
series and the layer at k Delta on the decimated series agree at the block
ends to 1e-8 (float64), state and output, through the gated block too. A
depthwise convolution breaks it (a filter in steps, not in physical time),
which is why the block ships without one.

## Experiment log

Every run of this project, with its prediction written before launch, its
thresholds and its verdict, is recorded in a dated experiment log kept outside
the repository. I am glad to share it with anyone who wants to know what
happened during training, including what did not work: open an issue or write
to the address in the citation.

## Citation

`CITATION.cff` at the root of the repository, or:

```bibtex
@misc{vincent2026timessm,
  author = {Vincent, Yvann},
  title  = {TimeSSM: a state space forecaster with a rate knob},
  year   = {2026},
  url    = {https://github.com/IUseAMouse/TimeSSM}
}
```

The sampling-rate equivariance and the frequency rule follow FlowState:

```bibtex
@article{graf2025flowstate,
  title   = {FlowState: Sampling Rate Invariant Time Series Forecasting},
  author  = {Graf, Lars and Ortner, Thomas and Wo{\'z}niak, Stanis{\l}aw and Pantazi, Angeliki},
  journal = {arXiv preprint arXiv:2508.05287},
  year    = {2025}
}
```
