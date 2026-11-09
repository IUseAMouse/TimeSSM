"""
External sources of the corpus, third batch: official statistics in bulk, river
gauges and water quality, weather stations, and more air monitors.

    python scripts/fetch_sources_c.py --out /workspace/data/v9_ext --raw /workspace/data/raw_d eurostat

Same layout and helpers as fetch_sources_b.py (cache under --raw, one manifest
entry per subset, a finished subset is left alone). The aim is the number of
distinct series: within a subset, series that are equal up to an affine map are
kept once. Licences and transformations are listed in docs/DATA_c.md.
"""

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_sources_b as fb
from fetch_sources_b import cached, done, emit, get

# Subsets of the earlier batches built from these tables or sites.
EUROSTAT_TAKEN = {"sts_inpr_m", "sts_trtu_m", "sts_copr_m", "namq_10_gdp", "nama_10_gdp",
                  "demo_r_mwk_ts"}
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination"
# Units that restate another series of the same table as a growth rate (rebased indices are
# affine copies, `distinct` drops them).
DERIVED_UNIT = r"PCH|^CH_|GROW"


def distinct(series):
    """Series equal up to an affine map (rebased index, change of unit) kept once."""
    seen, out = set(), []
    for x in series:
        x = fb.clean(x)
        if x is None:
            continue
        z = (x - np.nanmean(x)) / np.nanstd(x)
        key = hashlib.blake2b(np.round(np.nan_to_num(z, nan=99.0), 3).astype(np.float32).tobytes(),
                              digest_size=16).digest()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


# ------------------------------------------------------------ Eurostat
def _eurostat_catalogue():
    toc = pd.read_csv(io.BytesIO(get(f"{EUROSTAT_API}/catalogue/toc/txt?lang=en")), sep="\t")
    toc = toc[toc["type"].str.strip().isin(["dataset", "table"])].copy()
    toc["code"] = toc["code"].str.strip()
    start = toc["data start"].astype(str).str.strip()
    toc["f"] = np.select([start.str.contains(r"Q\d$"), start.str.fullmatch(r"\d{4}(M|-)\d\d"),
                          start.str.contains(r"W\d\d$")], ["Q", "M", "W"], "")
    return toc.drop_duplicates("code")


def _eurostat_table(code: str, freq: str):
    """Rows of one table at one rate: derived units and adjusted copies of an unadjusted
    series dropped, one series per remaining key."""
    path = cached(f"{EUROSTAT_API}/sdmx/2.1/data/{code}?format=TSV&compressed=true",
                  f"eurostat/{code}.tsv.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", dtype=str)
    head = df.columns[0]
    dims = head.split("\\")[0].split(",")
    keys = df[head].str.split(",", expand=True)
    keys.columns = dims
    times = [c for c in df.columns[1:] if c.strip()]
    pattern = {"M": r"^\d{4}-\d\d$", "Q": r"^\d{4}-Q\d$", "W": r"^\d{4}-W\d\d$"}[freq]
    times = sorted((c for c in times if re.match(pattern, c.strip())), key=str.strip)
    keep = keys["freq"].eq(freq) if "freq" in keys else pd.Series(True, index=keys.index)
    if not times or not keep.any():
        return []
    rank = pd.Series(0, index=keys.index)
    if "unit" in keys:
        rank += keys["unit"].str.contains(DERIVED_UNIT).astype(int) * 2
    if "s_adj" in keys:
        rank += keys["s_adj"].ne("NSA").astype(int)
    rest = [d for d in dims if d not in ("freq", "unit", "s_adj")]
    order = rank[keep].sort_values(kind="stable").index
    order = keys.loc[order].drop_duplicates(rest).index if rest else order[:1]
    # Cells read "12.3 p" (value and flags) or ": c" (missing).
    values = df.loc[order, times].apply(lambda c: pd.to_numeric(c.str.split().str[0], errors="coerce"))
    return list(values.to_numpy(np.float32))


def eurostat():
    """Every monthly and quarterly table of the catalogue, and weekly deaths by NUTS 3 region."""
    toc = _eurostat_catalogue()
    jobs = {"eurostat_bulk_m": ("M", toc.code[toc.f == "M"]),
            "eurostat_bulk_q": ("Q", toc.code[toc.f == "Q"]),
            "eurostat_deaths_nuts3_w": ("W", ["demo_r_mweek3"])}
    for subset, (freq, codes) in jobs.items():
        if done(subset):
            continue
        rows = []
        for code in codes:
            if code in EUROSTAT_TAKEN:
                continue
            try:
                part = _eurostat_table(code, freq)
            except Exception as e:
                print(code, repr(e)[:120], flush=True)
                continue
            rows.extend(part)
            print(code, len(part), flush=True)
        names = {"M": "MS", "Q": "Q-DEC", "W": "W-MON"}
        emit(subset, distinct(rows), names[freq], "official statistics",
             "Eurostat re-use policy (source acknowledged)",
             "https://ec.europa.eu/eurostat/web/main/data/database")


# ------------------------------------------------------------ USGS
USGS = "https://waterservices.usgs.gov/nwis"
STATES = ("al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh "
          "nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy pr").split()
# Discharge, water temperature, specific conductance, dissolved oxygen, pH, turbidity.
USGS_PARAMS = {"00060": ("usgs_discharge_15T", 2500), "00010": ("usgs_water_temp_15T", 800),
               "00095": ("usgs_conductance_15T", 600), "00300": ("usgs_oxygen_15T", 500),
               "00400": ("usgs_ph_15T", 400), "63680": ("usgs_turbidity_15T", 400)}


def _usgs_sites(param: str):
    sites = []
    for state in STATES:
        try:
            listing = get(f"{USGS}/site/?format=rdb&stateCd={state}&parameterCd={param}"
                          "&siteStatus=active&hasDataTypeCd=iv").decode()
        except Exception:
            continue
        sites += [ln.split("\t")[1] for ln in listing.splitlines() if ln.startswith("USGS\t")]
    return sorted(set(sites))


def _usgs_site(args):
    param, site = args
    path = fb.RAW / "usgs" / param / f"{site}.npy"
    if path.exists():
        return np.load(path)
    try:
        lines = get(f"{USGS}/iv/?format=rdb&parameterCd={param}&sites={site}"
                    "&startDT=2021-01-01&endDT=2024-12-31").decode().splitlines()
    except Exception:
        return None
    rows = [ln.split("\t") for ln in lines if ln.startswith("USGS\t")]
    if len(rows) < fb.MIN_OBS:
        return None
    # One value column per sensor; the first one is kept (rdb: agency, site, datetime, tz, value, flag).
    t = pd.to_datetime([r[2] for r in rows], errors="coerce")
    v = pd.to_numeric(pd.Series([r[4] if len(r) > 4 else None for r in rows]), errors="coerce")
    s = pd.Series(v.to_numpy(np.float32), index=t)
    s = s[s.index.notna()].resample("15min").mean()
    x = s.to_numpy(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def usgs():
    """Instantaneous values of river gauges, 2021 to 2024 on a 15-minute grid, a random sample
    of the active sites of each parameter; the sites of usgs_streamflow are left out."""
    taken = {p.stem for p in (fb.RAW.parent / "ext_raw" / "usgs").glob("*.rdb")}
    for param, (subset, n) in USGS_PARAMS.items():
        if done(subset):
            continue
        sites = [s for s in _usgs_sites(param) if s not in taken]
        rng = np.random.default_rng(0)
        picks = [sites[i] for i in rng.permutation(len(sites))[:n]]
        with cf.ThreadPoolExecutor(8) as pool:
            rows = [r for r in pool.map(_usgs_site, [(param, s) for s in picks]) if r is not None]
        emit(subset, rows, "15T", "hydrology", "Public domain (USGS)",
             "https://waterservices.usgs.gov/")


def usgs_more():
    """Second sample of 3,000 discharge sites, the sites of the first sample left out."""
    if done("usgs_discharge_15T_s2"):
        return
    taken = {p.stem for p in (fb.RAW.parent / "ext_raw" / "usgs").glob("*.rdb")}
    taken |= {p.stem for p in (fb.RAW / "usgs" / "00060").glob("*.npy")}
    sites = [s for s in _usgs_sites("00060") if s not in taken]
    rng = np.random.default_rng(1)
    picks = [sites[i] for i in rng.permutation(len(sites))[:3000]]
    with cf.ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(_usgs_site, [("00060", s) for s in picks]) if r is not None]
    emit("usgs_discharge_15T_s2", rows, "15T", "hydrology", "Public domain (USGS)",
         "https://waterservices.usgs.gov/")


# ------------------------------------------------------------ NOAA ISD-Lite
ISD = "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data"
ISD_CHANNELS = {0: "temp", 1: "dew", 2: "pressure", 4: "wind"}


def _isd_station(station: str):
    path = fb.RAW / "isd" / f"{station}.npy"
    if path.exists():
        return np.load(path)
    parts = []
    for year in range(2020, 2025):
        try:
            raw = gzip.decompress(get(f"{ISD}/{year}/{station}-{year}.gz", retries=2))
        except Exception:
            continue
        a = np.loadtxt(io.StringIO(raw.decode()), dtype=np.int64, ndmin=2)
        if len(a):
            parts.append(a)
    if not parts:
        return None
    a = np.concatenate(parts)
    hours = (pd.to_datetime(dict(year=a[:, 0], month=a[:, 1], day=a[:, 2], hour=a[:, 3]))
             .to_numpy().astype("datetime64[h]").astype(np.int64))
    hours -= hours.min()
    grid = np.full((len(ISD_CHANNELS), hours.max() + 1), np.nan, np.float32)
    for j, col in enumerate(ISD_CHANNELS):
        v = a[:, 4 + col].astype(np.float32)
        v[v == -9999] = np.nan
        grid[j, hours] = v / 10
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid)
    return grid


def isd(n: int = 3000, suffix: str = ""):
    """Hourly temperature, dew point, sea-level pressure and wind speed of a random sample of
    stations active from 2020 to 2024; Australian stations and Jena left out. A later sample
    (`suffix`) leaves out the stations already fetched."""
    if done(f"isd_temp_H{suffix}"):
        return
    taken = {p.stem for p in (fb.RAW / "isd").glob("*.npy")} if suffix else set()
    hist = pd.read_csv(cached("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"), dtype=str)
    ok = ((hist.BEGIN <= "20200101") & (hist.END >= "20241231") & (hist.CTRY != "AS")
          & ~hist["STATION NAME"].fillna("").str.contains("JENA") & (hist.USAF != "999999"))
    stations = sorted(set((hist.USAF + "-" + hist.WBAN)[ok]) - taken)
    rng = np.random.default_rng(len(suffix))
    picks = [stations[i] for i in rng.permutation(len(stations))[:n]]
    with cf.ThreadPoolExecutor(16) as pool:
        grids = [g for g in pool.map(_isd_station, picks) if g is not None]
    for j, name in enumerate(ISD_CHANNELS.values()):
        emit(f"isd_{name}_H{suffix}", [g[j] for g in grids], "H", "weather", "Public domain (NOAA NCEI)",
             "https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database")


def isd_more():
    """Second sample of 4,000 stations."""
    isd(4000, "_s2")


# ------------------------------------------------------------ NOAA NDBC buoys
NDBC = "https://www.ndbc.noaa.gov/data/historical/stdmet"
NDBC_CHANNELS = ("WSPD", "GST", "WVHT", "DPD", "APD", "PRES", "ATMP", "WTMP", "DEWP")


def _ndbc_station(station: str):
    path = fb.RAW / "ndbc" / f"{station}.npy"
    if path.exists():
        return np.load(path)
    parts = []
    for year in range(2020, 2025):
        try:
            text = gzip.decompress(get(f"{NDBC}/{station}h{year}.txt.gz", retries=2)).decode()
        except Exception:
            continue
        lines = text.splitlines()
        # Header, then a units line; some files repeat both further down.
        body = [line for line in lines[1:] if not line.startswith("#")]
        if not body:
            continue
        try:
            frame = pd.read_csv(io.StringIO("\n".join([lines[0].lstrip("#")] + body)),
                                sep=r"\s+", dtype=float)
        except ValueError:
            continue
        parts.append(frame)
    if not parts:
        return None
    frame = pd.concat(parts)
    stamp = pd.to_datetime(dict(year=frame["YY"], month=frame["MM"], day=frame["DD"],
                                hour=frame["hh"], minute=frame["mm"]), errors="coerce")
    cols = [c for c in NDBC_CHANNELS if c in frame]
    # Missing values are written as 99, 999 or 9999 in the column's own format.
    values = frame[cols].where(~frame[cols].isin([99.0, 999.0, 9999.0]))
    values.index = stamp
    grid = values[values.index.notna()].resample("h").mean().reindex(columns=list(NDBC_CHANNELS))
    x = grid.to_numpy(np.float32).T
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def ndbc():
    """Buoys and coastal stations of the NDBC, 2020 to 2024: wind speed and gust, wave height,
    dominant and average wave period, pressure, air, water and dew point temperature, means
    over the hour."""
    if done("ndbc_wspd_H"):
        return
    listing = get(f"{NDBC}/").decode()
    stations = sorted(set(re.findall(r'href="([a-z0-9]+)h202[0-4]\.txt\.gz"', listing)))
    with cf.ThreadPoolExecutor(8) as pool:
        grids = [g for g in pool.map(_ndbc_station, stations) if g is not None]
    for j, name in enumerate(NDBC_CHANNELS):
        emit(f"ndbc_{name.lower()}_H", [g[j] for g in grids], "H", "ocean",
             "Public domain (NOAA NDBC)", "https://www.ndbc.noaa.gov/")


# ------------------------------------------------------------ EPA, more parameters
EPA_FILES = {"42101": "epa_co", "42401": "epa_so2", "81102": "epa_pm10", "WIND": "epa_wind",
             "TEMP": "epa_temp", "PRESS": "epa_pressure", "RH_DP": "epa_humidity"}


def epa_more():
    """Hourly carbon monoxide, SO2, PM10 and the meteorology of the AQS monitors, 2021 to
    2024, one series per monitor and parameter."""
    usecols = ["State Code", "County Code", "Site Num", "Parameter Code", "POC", "Date GMT",
               "Time GMT", "Sample Measurement"]
    for code, subset in EPA_FILES.items():
        if done(subset):
            continue
        parts = []
        for year in range(2021, 2025):
            z = zipfile.ZipFile(cached(f"https://aqs.epa.gov/aqsweb/airdata/hourly_{code}_{year}.zip"))
            with z.open(z.namelist()[0]) as f:
                parts.append(pd.read_csv(f, usecols=usecols, dtype=str))
        df = pd.concat(parts)
        df["t"] = pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"])
        df["v"] = pd.to_numeric(df["Sample Measurement"], errors="coerce")
        df = df[df["Parameter Code"] != "61104"]          # wind direction, an angle
        keys = ["State Code", "County Code", "Site Num", "Parameter Code", "POC"]
        rows = [fb.regular(g.set_index("t")[["v"]], "h")["v"].to_numpy()
                for _, g in df.groupby(keys, sort=False)]
        emit(subset, rows, "H", "air quality", "Public domain (US EPA)",
             "https://aqs.epa.gov/aqsweb/airdata/download_files.html")


FETCHERS = {f.__name__: f for f in (eurostat, usgs, isd, epa_more, usgs_more, isd_more, ndbc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", default=list(FETCHERS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    a = ap.parse_args()
    fb.RAW, fb.OUT = a.raw, a.out
    a.raw.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.sources:
        try:
            FETCHERS[name]()
        except Exception as e:
            print(f"{name} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
