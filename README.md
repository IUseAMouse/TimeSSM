![TimeSSM: a 10M-parameter forecaster that chooses its sampling rate](docs/assets/timessm_banner.png)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

TimeSSM is a family of small zero-shot time series forecasters built on diagonal linear
state space layers (S4D). The sampling rate is part of the model: a scalar tells the trunk
which rate it is reading, and the RateIN layer, which has no learned weights, chooses at
inference the rates worth reading from the past of the series. The models output nine
quantiles for any horizon in one pass.

| Model | Parameters | GIFT-Eval CRPS | GIFT-Eval MASE | Weights |
|---|---|---|---|---|
| **TimeSSM-10M** | 9.98M | **0.4763** | **0.6948** | [IUseAMouse/timessm-10m](https://huggingface.co/IUseAMouse/timessm-10m) |
| TimeSSM-2.5M | 2.5M | 0.5128 | 0.7544 | [IUseAMouse/timessm-2.5m](https://huggingface.co/IUseAMouse/timessm-2.5m) |

GIFT-Eval, 97 configurations, zero-shot, geometric means of the ratio to the official
seasonal naive; lower is better. Both models were trained from scratch by one person,
outside of any lab or company: the 10M in 31 hours on one A100, the 2.5M on three RTX 3090s.

## TimeSSM-10M

Four pairs of a causal S4D layer and causal attention over patches of 16 steps, applied in
two shared passes; a context of up to 4096 steps; 9,982,512 parameters. Trained on 13.3
billion points of public and synthetic series from which every series found in GIFT-Eval
was removed, by name and by value, and selected on held-out subsets of that corpus, never on
GIFT-Eval. The model card ([docs/model_card_timessm_10m.md](docs/model_card_timessm_10m.md))
has the 97 per-configuration numbers; the technical report describes the model, its training
and what did not work.

Among the entries of public size under 10M parameters in the official results folder on
12 November 2026, recomputed from their per-configuration results:

| Model | Parameters | CRPS | MASE |
|---|---|---|---|
| PDM-Femto | 6M | 0.4676 | 0.6843 |
| **TimeSSM-10M** | **9.98M** | **0.4763** | **0.6948** |
| Timestral-5M | 5M | 0.4789 | 0.7049 |
| FlowState-9.1M | 9.1M | 0.5019 | 0.7262 |
| TimeSSM-2.5M | 2.5M | 0.5128 | 0.7544 |
| TTM-R3 | 1.4M | 0.5195 | 0.7240 |
| Toto-2.0-4m | 4M | 0.5242 | 0.7565 |

TTM-R3 saw GIFT-Eval data in pretraining. For scale, Chronos-2 (120M) is at 0.4854 and
TimesFM-2.5 (200M) at 0.4903. The model is at its best on series sampled every 5 to 15
minutes, and behind models of its size on daily and weekly series.

## How to use

```bash
pip install git+https://github.com/IUseAMouse/TimeSSM.git
```

```python
import numpy as np
import torch
from timessm import from_pretrained, delta_scale

model = from_pretrained("IUseAMouse/timessm-10m")   # or "IUseAMouse/timessm-2.5m", or a local directory

series = np.sin(np.arange(1024) * 2 * np.pi / 24)   # your history: raw values, any scale
context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)   # [1, length]
w = delta_scale(24, *model.delta_range)   # season length in steps: 24 for hourly data with a daily cycle
with torch.no_grad():
    out = model.forecast(context, 48, w=w)
quantiles = out["quantiles_denorm"][0]   # [48, 9], levels 0.1 to 0.9; column 4 is the median
```

The first argument of `delta_scale` is the season length of the declared frequency, in
steps: 96 for 15-minute data, 24 for hourly, 7 for daily data with a weekly cycle, 12 for
monthly. Normalization lives inside the model: feed raw values, read raw values back. This
call runs the trunk at one rate; the full model, with the RateIN layer, runs through the
evaluation harness below.

## Reproduce the numbers

Training and the GIFT-Eval harness use the `timejepa` package, behind the `train` extra (a
checkout next to this repository, `../TimeJEPA`, is used when present).

```bash
git clone -b timessm-10m https://github.com/IUseAMouse/TimeJEPA && git clone -b timessm-10m https://github.com/IUseAMouse/TimeSSM
cd TimeJEPA && make gift-download && cd ../TimeSSM && uv sync --extra train --extra dev
uv run pytest -q

hf download IUseAMouse/timessm-10m timessm_10m.ckpt --local-dir .
EVAL_CONFIG=timessm_10m_gift scripts/eval_ssm.sh timessm_10m.ckpt                # MASE 0.6948, CRPS 0.4763

hf download IUseAMouse/timessm-2.5m epoch00_valloss1.2814.ckpt --local-dir .
EVAL_CONFIG=timessm_2.5m_gift scripts/eval_ssm.sh epoch00_valloss1.2814.ckpt     # MASE 0.7544, CRPS 0.5128
```

Each config carries every inference setting of its number. Training the 10M:

```bash
scripts/train_patched_loop.sh configs/patched/final.yaml run.name=timessm_10m
```

The corpus builders are `scripts/build_*`; [docs/DATA.md](docs/DATA.md) describes the corpus
and how the overlap with GIFT-Eval is checked by value.

## RateIN

Forecasting models have a preferred sampling regime, and presenting a series at that regime
improves their forecasts. RateIN ([paper](https://doi.org/10.5281/zenodo.23868904)) reads the
context at several rates (coarser by 2 to 48, finer by 2 to 4), forecasts at each, weights the
rates by a causal backtest on the past of the series, and mixes the forecasts per quantile
level, with sign-flip averaging. Nothing is learned and nothing is fitted on the test data.
A state space layer suits it: a continuous-time linear system discretized with a zero-order
hold reads a series decimated by k at step Delta exactly as it reads the full series at
Delta / k, which is the convention of FlowState ([Graf et al., 2026](https://arxiv.org/abs/2508.05287)).

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

The cost is one forecast per rate and per sign, and a backtest on the past of the series.

## Layout

```
src/timessm/patched.py    PatchedSSM (TimeSSM-10M): causal normalization, S4D + attention pairs on patches,
                          shared passes, quantile head; forecast(x, n, w)
src/timessm/corpus.py     training windows: rate classes, decimation, sign flip, full-history windows
src/timessm/model.py      SSMForecaster (TimeSSM-2.5M): one token per step, gated S4D blocks
src/timessm/ssm.py        S4DLayer: ZOH discretization, FFT convolution, step recurrence
src/timessm/hub.py        from_pretrained
scripts/train_patched.py  training of the 10M (configs/patched/); scripts/train_ssm.py for the 2.5M
scripts/dev_eval.py       model selection on held-out subsets of the corpus
scripts/eval_ssm.sh       GIFT-Eval through the harness
configs/timessm_10m_gift.yaml, configs/timessm_2.5m_gift.yaml   the settings of the released numbers
docs/                     model cards, corpus, hypotheses and their verdicts, every look at GIFT-Eval
legacy/                   the March 2025 Mamba code this repository started from, untouched
```

`patched.py` also holds an attention across the variates of a multivariate item, with past and
future covariates. A model trained with it did not beat the released one and is not released;
the code and its tests are kept.

## Rate equivariance, tested

`tests/test_ssm_layer.py` pins the property before anything else: on a
block-constant input (blocks of k steps) the layer at Delta on the fine
series and the layer at k Delta on the decimated series agree at the block
ends to 1e-8 (float64), state and output, through the gated block too. A
depthwise convolution breaks it (a filter in steps, not in physical time),
which is why the block ships without one.

## Citation

`CITATION.cff` at the root of the repository, or:

```bibtex
@misc{vincent2026timessm,
  author = {Vincent, Yvann},
  title  = {TimeSSM: a state space forecaster with a rate knob},
  year   = {2026},
  url    = {https://github.com/IUseAMouse/TimeSSM}
}

@misc{vincent_2026_23868904,
  author    = {Vincent, Yvann},
  title     = {RateIN: Time Series Foundation Models Have a Sampling Problem},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23868904}
}
```

The sampling-rate equivariance and the frequency rule follow FlowState:

```bibtex
@inproceedings{graf2026flowstate,
  title     = {FlowState: Sampling-Rate-Equivariant Time-Series Forecasting},
  author    = {Graf, Lars and Ortner, Thomas and Wo{\'z}niak, Stanis{\l}aw and Pantazi, Angeliki},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {306},
  year      = {2026},
  note      = {arXiv:2508.05287}
}
```
