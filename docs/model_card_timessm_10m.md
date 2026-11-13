---
license: apache-2.0
library_name: timessm
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - zero-shot
  - state-space-model
  - gift-eval
model-index:
  - name: TimeSSM-10M
    results:
      - task:
          type: time-series-forecasting
        dataset:
          name: GIFT-Eval (97 tasks)
          type: Salesforce/GiftEval
        metrics:
          - type: mase
            name: MASE (normalised by Seasonal Naive)
            value: 0.6948
          - type: crps
            name: CRPS (normalised by Seasonal Naive)
            value: 0.4763
---

![TimeSSM: a 10M-parameter forecaster that chooses its sampling rate](assets/timessm_banner.png)

# TimeSSM-10M

A zero-shot time series forecaster of 9.98 million parameters. Its trunk interleaves diagonal
linear state space layers (S4D) with attention over patches of 16 steps, reads up to 4096 steps
and outputs nine quantiles for any horizon in one pass. The sampling rate is part of the model:
a scalar tells the trunk which rate it is reading, and the RateIN layer, which has no learned
weights, chooses at inference the rates worth reading from the past of the series.

It was trained from scratch in 31 hours on one A100, by one person, outside of any lab or
company. A technical report will follow.

## GIFT-Eval

97 configurations, zero-shot. MASE and CRPS are geometric means of the ratio to the official
seasonal naive, as on the leaderboard. Lower is better.

| | MASE | CRPS |
|---|---|---|
| **TimeSSM-10M** | **0.6948** | **0.4763** |

The 80 % interval covers 79.3 % of the targets. Every series found in GIFT-Eval was removed from
the training corpus, by name and by value. The model and its settings were selected on held-out
subsets of the corpus, never on GIFT-Eval.

### Under 10 million parameters

Entries of public size in the official results folder on 12 November 2026, recomputed from their
per-configuration results.

| Model | Parameters | CRPS | MASE | Availability |
|---|---|---|---|---|
| PDM-Femto | 6M | 0.4676 | 0.6843 | API only |
| **TimeSSM-10M** | **9.98M** | **0.4763** | **0.6948** | Apache-2.0 |
| Timestral-5M | 5M | 0.4789 | 0.7049 | weights, modified MIT |
| FlowState-9.1M | 9.1M | 0.5019 | 0.7262 | open weights |
| TimeSSM-2.5M | 2.5M | 0.5128 | 0.7544 | Apache-2.0 |
| TTM-R3 | 1.4M | 0.5195 | 0.7240 | open weights |
| Toto-2.0-4m | 4M | 0.5242 | 0.7565 | open weights |

TTM-R3 saw GIFT-Eval data in pretraining; 1.4M is the size of its main checkpoint.

### Larger models, for scale

| Model | Parameters | CRPS | MASE |
|---|---|---|---|
| Timestral-28M | 28M | 0.4610 | 0.6780 |
| t0-beta | 256M | 0.4738 | 0.6865 |
| Toto-2.0-2.5B | 2.5B | 0.4759 | 0.6956 |
| TiRex-2-Zeroshot | 82.5M | 0.4781 | 0.6973 |
| Chronos-2 | 120M | 0.4854 | 0.6978 |
| TimesFM-2.5 | 200M | 0.4903 | 0.7050 |

### By sampling rate

| Rate | Configurations | CRPS |
|---|---|---|
| 10 seconds | 6 | 0.547 |
| 5 to 15 minutes | 30 | 0.468 |
| hourly | 31 | 0.426 |
| daily | 15 | 0.422 |
| weekly | 8 | 0.620 |
| monthly | 5 | 0.699 |
| quarterly | 1 | 0.736 |
| yearly | 1 | 0.788 |

The model is at its best on series sampled every 5 to 15 minutes, and at its weakest, against
models of its size, on daily and weekly series.

<details>
<summary>97 configurations</summary>

| Configuration | MASE | CRPS |
|---|---|---|
| bitbrains_fast_storage/H/short | 0.764 | 0.627 |
| solar/H/short | 0.884 | 0.526 |
| solar/H/medium | 0.932 | 0.341 |
| solar/H/long | 0.851 | 0.299 |
| m_dense/D/short | 0.386 | 0.269 |
| ett2/H/short | 0.791 | 0.708 |
| ett2/H/medium | 0.845 | 0.547 |
| ett2/H/long | 0.933 | 0.493 |
| ett2/D/short | 0.985 | 0.650 |
| electricity/15T/short | 0.583 | 0.515 |
| electricity/15T/medium | 0.715 | 0.646 |
| electricity/15T/long | 0.742 | 0.639 |
| bitbrains_rnd/H/short | 0.988 | 0.502 |
| loop_seattle/H/short | 0.623 | 0.537 |
| loop_seattle/H/medium | 0.578 | 0.362 |
| loop_seattle/H/long | 0.545 | 0.306 |
| solar/W/short | 0.919 | 0.873 |
| bizitobs_application/10S/short | 0.456 | 0.285 |
| bizitobs_application/10S/medium | 0.790 | 0.770 |
| bizitobs_application/10S/long | 0.942 | 1.087 |
| bizitobs_service/10S/short | 0.615 | 0.271 |
| bizitobs_service/10S/medium | 0.779 | 0.438 |
| bizitobs_service/10S/long | 0.943 | 0.945 |
| bizitobs_l2c/5T/short | 0.269 | 0.268 |
| bizitobs_l2c/5T/medium | 0.362 | 0.357 |
| bizitobs_l2c/5T/long | 0.315 | 0.307 |
| ett2/W/short | 0.870 | 0.751 |
| loop_seattle/5T/short | 0.733 | 0.595 |
| loop_seattle/5T/medium | 0.622 | 0.539 |
| loop_seattle/5T/long | 0.603 | 0.526 |
| loop_seattle/D/short | 0.513 | 0.418 |
| bitbrains_rnd/5T/short | 0.877 | 0.367 |
| bitbrains_rnd/5T/medium | 0.992 | 0.526 |
| bitbrains_rnd/5T/long | 0.978 | 0.517 |
| electricity/W/short | 0.676 | 0.479 |
| solar/D/short | 0.858 | 0.533 |
| ett1/H/short | 0.839 | 0.744 |
| ett1/H/medium | 0.835 | 0.610 |
| ett1/H/long | 0.950 | 0.597 |
| temperature_rain/D/short | 0.671 | 0.437 |
| saugeen/W/short | 0.593 | 0.480 |
| covid_deaths/D/short | 0.670 | 0.293 |
| us_births/D/short | 0.196 | 0.159 |
| m4_weekly/W/short | 0.737 | 0.593 |
| car_parts/M/short | 0.725 | 0.576 |
| sz_taxi/H/short | 0.767 | 0.640 |
| ett1/15T/short | 0.724 | 0.648 |
| ett1/15T/medium | 0.834 | 0.712 |
| ett1/15T/long | 0.833 | 0.666 |
| saugeen/M/short | 0.729 | 0.634 |
| us_births/M/short | 0.845 | 0.780 |
| m4_quarterly/Q/short | 0.702 | 0.736 |
| ett1/W/short | 0.917 | 0.904 |
| saugeen/D/short | 0.795 | 0.551 |
| electricity/H/short | 0.691 | 0.621 |
| electricity/H/medium | 0.785 | 0.611 |
| electricity/H/long | 0.780 | 0.562 |
| ett1/D/short | 0.953 | 0.689 |
| bizitobs_l2c/H/short | 0.381 | 0.371 |
| bizitobs_l2c/H/medium | 0.314 | 0.245 |
| bizitobs_l2c/H/long | 0.386 | 0.264 |
| hospital/M/short | 0.813 | 0.797 |
| m4_daily/D/short | 0.981 | 0.861 |
| bitbrains_fast_storage/5T/short | 0.646 | 0.305 |
| bitbrains_fast_storage/5T/medium | 0.865 | 0.539 |
| bitbrains_fast_storage/5T/long | 0.825 | 0.580 |
| ett2/15T/short | 0.673 | 0.643 |
| ett2/15T/medium | 0.787 | 0.674 |
| ett2/15T/long | 0.845 | 0.659 |
| jena_weather/10T/short | 0.362 | 0.177 |
| jena_weather/10T/medium | 0.809 | 0.224 |
| jena_weather/10T/long | 0.820 | 0.206 |
| jena_weather/D/short | 0.740 | 0.240 |
| us_births/W/short | 0.678 | 0.640 |
| m_dense/H/short | 0.528 | 0.471 |
| m_dense/H/medium | 0.435 | 0.300 |
| m_dense/H/long | 0.465 | 0.270 |
| kdd_cup_2018/D/short | 0.789 | 0.555 |
| electricity/D/short | 0.710 | 0.536 |
| restaurant/D/short | 0.689 | 0.384 |
| jena_weather/H/short | 0.733 | 0.273 |
| jena_weather/H/medium | 0.888 | 0.154 |
| jena_weather/H/long | 0.668 | 0.131 |
| m4_monthly/M/short | 0.718 | 0.735 |
| hierarchical_sales/D/short | 0.666 | 0.335 |
| m4_hourly/H/short | 0.601 | 0.518 |
| m4_yearly/A/short | 0.781 | 0.788 |
| hierarchical_sales/W/short | 0.706 | 0.423 |
| kdd_cup_2018/H/short | 0.708 | 0.692 |
| kdd_cup_2018/H/medium | 0.735 | 0.555 |
| kdd_cup_2018/H/long | 0.760 | 0.469 |
| sz_taxi/15T/short | 0.711 | 0.651 |
| sz_taxi/15T/medium | 0.757 | 0.546 |
| sz_taxi/15T/long | 0.750 | 0.479 |
| solar/10T/short | 0.780 | 0.508 |
| solar/10T/medium | 0.865 | 0.466 |
| solar/10T/long | 0.907 | 0.439 |

</details>

## Model

Four layers, each a causal S4D layer followed by causal attention over the patch
tokens, applied in two shared passes; width 352, state size 32, 8 heads; 9,982,512 parameters.
Each series is normalized causally, patch by patch, then passed through an arcsinh. The horizon is
read from learned future tokens, so there is no autoregressive loop. A scalar `w` scales the
time step of the state space layers and the positions of the attention: `w = 24 / season`, with
`season` the number of steps in a cycle of the declared frequency.

RateIN ([paper](https://doi.org/10.5281/zenodo.23868904)) reads the context at several rates
(coarser by 2 to 48, finer by 2 to 4), scores each on a causal backtest over the past of the
series, and mixes the forecasts of the best rates, with sign-flip averaging. It learns nothing
and fits nothing on the evaluation data.

Training: pinball loss on nine quantiles, on every context token as well as on the horizon;
13.3 billion points of public and synthetic series; 200,000 steps with a constant learning rate
then a linear decay over the last 40,000.

## Inference

```bash
pip install git+https://github.com/IUseAMouse/TimeSSM.git@timessm-10m
```

```python
import numpy as np
import torch
from timessm import from_pretrained, delta_scale

model = from_pretrained("IUseAMouse/timessm-10m")   # or a local directory; device="cuda" to run there

series = np.sin(np.arange(1024) * 2 * np.pi / 24)   # your history: raw values, any scale, NaN allowed
context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)   # [1, length]
w = delta_scale(24, *model.delta_range)   # season length in steps: 24 for hourly data with a daily cycle
with torch.no_grad():
    out = model.forecast(context, 48, w=w)
quantiles = out["quantiles_denorm"][0]   # [48, 9], levels 0.1 to 0.9; column 4 is the median
```

This call runs the trunk at one rate. The full model, with the RateIN layer, runs through the
evaluation harness (next section).

## Reproduction

```bash
git clone -b timessm-10m https://github.com/IUseAMouse/TimeJEPA && git clone -b timessm-10m https://github.com/IUseAMouse/TimeSSM
cd TimeJEPA && make gift-download && cd ../TimeSSM && uv sync --extra train && source .venv/bin/activate
hf download IUseAMouse/timessm-10m timessm_10m.ckpt --local-dir .
EVAL_CONFIG=timessm_10m_gift scripts/eval_ssm.sh timessm_10m.ckpt     # MASE 0.6948, CRPS 0.4763
```

These commands reproduce the number above from the released weights; the last one takes about
four hours on one A100. The tag `timessm-10m` of both repositories is the code that produced it.
`configs/timessm_10m_gift.yaml` carries every inference setting. The harness writes
`all_results.csv` in the leaderboard's format and recomputes the metrics from the raw data with
the official seasonal-naive denominators.

## Files and license

- `model.safetensors` and `config.json`, read by `timessm.from_pretrained`; `timessm_10m.ckpt`,
  the same weights in the format of the harness.
- Code: [TimeSSM](https://github.com/IUseAMouse/TimeSSM) (Apache-2.0) and
  [TimeJEPA](https://github.com/IUseAMouse/TimeJEPA) (harness, MIT).
- Weights: Apache-2.0.

## Citation

```bibtex
@misc{vincent2026timessm10m,
  author = {Vincent, Yvann},
  title  = {TimeSSM-10M: a small forecaster that chooses its sampling rate},
  year   = {2026},
  url    = {soon}
}

@misc{vincent_2026_23868904,
  author    = {Vincent, Yvann},
  title     = {RateIN: Time Series Foundation Models Have a Sampling Problem},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23868904}
}
```
