"""TimeSSM: a linear time-invariant state space forecaster with a rate knob.

Inference needs only torch: `build_from_config`, `load_checkpoint` and
`SSMForecaster.forecast`. Training and the GIFT-Eval harness use the
`timejepa` package (the `train` extra).
"""

from .ssm import S4DLayer
from .block import GatedSSMBlock
from .model import SSMForecaster, build_from_config
from .loading import load_checkpoint
from .hub import from_pretrained
from .frequency import BASE_SEASON, delta_scale, season_length

__all__ = ["S4DLayer", "GatedSSMBlock", "SSMForecaster", "build_from_config",
           "load_checkpoint", "from_pretrained", "BASE_SEASON", "delta_scale", "season_length"]
