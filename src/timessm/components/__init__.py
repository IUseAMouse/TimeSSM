"""Input normalization, patching and the quantile head, copied from TimeJEPA
(src/timejepa/models/components and models/decoders/linear_decoder) so that
inference needs nothing but torch. Attribute names are unchanged: a
checkpoint trained with the TimeJEPA classes loads as it is.
"""

from .patching import Patching, UnPatching
from .revin import RevIN
from .robust_scale import RobustScale
from .forecasting_head import ForecastingHead

__all__ = ["Patching", "UnPatching", "RevIN", "RobustScale", "ForecastingHead"]
