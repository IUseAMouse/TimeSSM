"""
From a declared sampling frequency to a rate scale.

The rate scale is w = BASE_SEASON / season, where `season` is the number of
steps of the series' reference cycle, so one cycle spans the same model time
whatever the sampling rate. This is FlowState's convention; the mapping below
follows its reference implementation
(`tsfm_public/models/flowstate/utils/utils.py::get_fixed_factor` in
ibm-granite/granite-tsfm):

    seconds   one hour     3600 / n           10S -> 360
    minutes   one day      1440 / n           15T -> 96
    hours     one day      24 / n             6H  -> 4
    days      one week     7 / n   if the domain follows a human rhythm
              one year     365 / n otherwise
    weeks     one year     365 / 7 / n
    months    one year     12 / n
    quarters  one year     4 / n
    years     4            (their choice for annual data)

`weekly` is FlowState's rule for daily data: only Transport, Healthcare and
Sales have a weekly cycle. Their GIFT wrapper adds one exception, kept here:
bizitobs_l2c has a weekly cycle and no daily one, so its season is 7 times longer.

`gift.DEFAULT_SEASONALITIES` (the gluonts table behind the MASE denominator)
is unrelated to this module.
"""

import re
from pathlib import Path
from typing import Dict, Optional

BASE_SEASON = 24.0
PER_ROW = -1.0      # table value: the season of each row is in the corpus sidecar _season/<file>.npy
WEEKLY_DOMAINS = ("Transport", "Healthcare", "Sales")

_UNITS = {
    "s": "S", "sec": "S",
    "t": "T", "min": "T",
    "h": "H",
    "d": "D", "b": "D",
    "w": "W",
    "m": "M", "me": "M", "ms": "M",
    "q": "Q", "qe": "Q", "qs": "Q",
    "a": "A", "y": "A", "ye": "A", "ys": "A", "as": "A",
}
_CYCLE = {"S": 3600.0, "T": 1440.0, "H": 24.0, "W": 365.0 / 7.0, "M": 12.0, "Q": 4.0}

# Domain of each GIFT-Eval dataset, from the official results files (the
# `domain` column of the leaderboard's all_results.csv).
GIFT_DOMAINS = {
    "bitbrains_fast_storage": "Web/CloudOps", "bitbrains_rnd": "Web/CloudOps",
    "bizitobs_application": "Web/CloudOps", "bizitobs_l2c": "Web/CloudOps",
    "bizitobs_service": "Web/CloudOps", "car_parts": "Sales", "covid_deaths": "Healthcare",
    "electricity": "Energy", "ett1": "Energy", "ett2": "Energy", "hierarchical_sales": "Sales",
    "hospital": "Healthcare", "jena_weather": "Nature", "kdd_cup_2018": "Nature",
    "loop_seattle": "Transport", "m4_daily": "Econ/Fin", "m4_hourly": "Econ/Fin",
    "m4_monthly": "Econ/Fin", "m4_quarterly": "Econ/Fin", "m4_weekly": "Econ/Fin",
    "m4_yearly": "Econ/Fin", "m_dense": "Transport", "restaurant": "Sales", "saugeen": "Nature",
    "solar": "Energy", "sz_taxi": "Transport", "temperature_rain": "Nature", "us_births": "Healthcare",
}


def parse_frequency(freq: str) -> Optional[tuple]:
    """'15T' -> ('T', 15); '15min' -> ('T', 15); 'h' -> ('H', 1); 'W-SUN' ->
    ('W', 1); 'QE-DEC' -> ('Q', 1). None when the unit is not recognized."""
    m = re.fullmatch(r"\s*(\d*)\s*([A-Za-z]+)(?:-[A-Za-z]+)?\s*", str(freq))
    if not m:
        return None
    unit = _UNITS.get(m.group(2).lower())
    return (unit, int(m.group(1) or 1)) if unit else None


def season_length(freq: Optional[str], weekly: bool = False) -> Optional[float]:
    """Steps of the reference cycle of a series sampled at `freq`; None when
    the frequency is missing or not recognized. `weekly` only matters for
    daily data (see the module docstring)."""
    parsed = parse_frequency(freq) if freq else None
    if parsed is None:
        return None
    unit, n = parsed
    if unit == "A":
        return 4.0
    if unit == "D":
        return (7.0 if weekly else 365.0) / n
    return _CYCLE[unit] / n


def delta_scale(season: float, lo: float, hi: float) -> float:
    """w = BASE_SEASON / season, clamped to the range the model was trained on."""
    return min(max(BASE_SEASON / float(season), lo), hi)


def gift_season(config: str) -> Optional[float]:
    """Season of a GIFT-Eval config 'dataset/freq/term', with the weekly rule
    from the dataset's official domain and the bizitobs_l2c exception."""
    dataset, freq = config.split("/")[:2]
    season = season_length(freq, weekly=GIFT_DOMAINS.get(dataset) in WEEKLY_DOMAINS)
    if season is not None and "l2c" in dataset:
        season *= 7.0
    return season


def load_frequency_table(path) -> Dict[str, float]:
    """A corpus frequency table (YAML): file stem -> season in steps, 0.0 for a
    file declared without a frequency (`null`), PER_ROW (-1.0) for a file whose
    rows carry their own season in the `_season/` sidecar.

        beijing_air_quality: {freq: H}
        favorita_sales:      {freq: D, weekly: true}
        some_file:           {season: 96}         # explicit, overrides freq
        synthetic_lowfreq:   null                 # no frequency: the random Delta draw
        synthetic_subhourly: {season: per_row}    # scripts/build_season_sidecars.py
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    table = {}
    for stem, entry in raw.items():
        if entry is None:
            table[stem] = 0.0
            continue
        if not isinstance(entry, dict):
            entry = {"freq": entry}
        if not isinstance(entry.get("weekly", False), bool):
            raise ValueError(f"{path}: {stem} has weekly: {entry['weekly']!r}; decide true or "
                             "false (does the series follow a weekly rhythm?)")
        season = entry.get("season")
        if season == "per_row":
            table[stem] = PER_ROW
            continue
        if season is None:
            season = season_length(entry.get("freq"), weekly=bool(entry.get("weekly", False)))
        if season is None or float(season) <= 0:
            raise ValueError(f"{path}: {stem} has no usable frequency ({entry}); write `null` "
                             "to declare a file without one")
        table[stem] = float(season)
    return table
