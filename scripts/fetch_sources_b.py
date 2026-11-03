"""
External sources of the corpus, second batch: air quality, road traffic, wind,
photovoltaics, machines, networks, and weekly to yearly statistics.

    python scripts/fetch_sources_b.py --out /workspace/data/v4_ext_b --raw /workspace/data/raw_b epa wind

Same layout as build_corpus_v4.py, one manifest entry per subset. Each fetcher
is rerunnable: downloads are cached under --raw and a finished subset is left
alone. Licences and transformations are listed in docs/DATA_b.md.
"""

import argparse
import concurrent.futures as cf
import fcntl
import io
import json
import re
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UA = {"User-Agent": "timessm-corpus/1.0 (https://github.com/IUseAMouse/TimeSSM)"}
MIN_OBS = 16
RAW = OUT = None


def get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def cached(url: str, name: str = None) -> Path:
    path = RAW / (name or url.rsplit("/", 1)[-1].split("?")[0])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, \
                tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


def done(subset: str) -> bool:
    return (OUT / f"{subset}.offsets.npy").exists()


def clean(x) -> np.ndarray:
    """Finite span of a series, constants dropped; None when too short."""
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, np.nan)
    obs = np.flatnonzero(np.isfinite(x))
    if len(obs) < MIN_OBS:
        return None
    x = x[obs[0]:obs[-1] + 1]
    return None if np.nanstd(x) == 0 else x


def emit(subset: str, series, freq: str, domain: str, license_: str, origin: str) -> None:
    rows = [s for s in (clean(x) for x in series) if s is not None]
    if not rows:
        print(f"{subset}: nothing kept", flush=True)
        return
    lengths = [len(r) for r in rows]
    np.save(OUT / f"{subset}.values.npy", np.concatenate(rows))
    np.save(OUT / f"{subset}.offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    entry = {"subset": subset, "n_series": len(rows), "n_obs": int(sum(lengths)), "freq": freq,
             "median_length": int(np.median(lengths)), "domain": domain, "license": license_,
             "origin": origin}
    path = OUT / "manifest.json"
    with (OUT / ".lock").open("w") as lock:        # several fetchers run at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest[subset] = entry
        path.write_text(json.dumps(manifest, indent=1))
    print(entry, flush=True)


def regular(frame: pd.DataFrame, step: str) -> pd.DataFrame:
    """Numeric columns of a time-indexed frame on a regular grid; holes stay NaN."""
    frame = frame[~frame.index.duplicated()].sort_index()
    return frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq=step))


def epa():
    """Hourly PM2.5, ozone and NO2, one series per monitor, 2019 to 2024."""
    for code, label in (("88101", "pm25"), ("44201", "ozone"), ("42602", "no2")):
        subset = f"epa_{label}"
        if done(subset):
            continue
        parts = []
        for year in range(2019, 2025):
            z = zipfile.ZipFile(cached(f"https://aqs.epa.gov/aqsweb/airdata/hourly_{code}_{year}.zip"))
            usecols = ["State Code", "County Code", "Site Num", "POC", "Date GMT", "Time GMT",
                       "Sample Measurement"]
            with z.open(z.namelist()[0]) as f:
                parts.append(pd.read_csv(f, usecols=usecols, dtype=str))
        df = pd.concat(parts)
        df["t"] = pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"])
        df["v"] = pd.to_numeric(df["Sample Measurement"], errors="coerce")
        df["id"] = df["State Code"] + df["County Code"] + df["Site Num"] + df["POC"]
        rows = []
        for _, g in df.groupby("id", sort=False):
            s = g.set_index("t")["v"]
            rows.append(regular(s.to_frame(), "h")["v"].to_numpy())
        emit(subset, rows, "H", "air quality", "Public domain (US EPA)",
             "https://aqs.epa.gov/aqsweb/airdata/download_files.html")


def _scada(paths, subset_prefix: str, origin: str):
    columns = {"Wind speed (m/s)": "wind", "Power (kW)": "power",
               "Nacelle ambient temperature (°C)": "temp", "Generator RPM (RPM)": "rpm"}
    series = {c: {} for c in columns.values()}
    for path in paths:
        z = zipfile.ZipFile(path)
        for name in z.namelist():
            if "Turbine_Data" not in name or not name.endswith(".csv"):
                continue
            turbine = re.search(r"Turbine_Data_\D*?(\d+)_", name) or re.search(r"_(\d+)_\d{4}", name)
            with z.open(name) as f:
                lines = [ln for ln in io.TextIOWrapper(f, "utf-8", errors="replace")]
            head = next(i for i, ln in enumerate(lines) if ln.startswith("# Date and time"))
            frame = pd.read_csv(io.StringIO("".join(lines[head:])), low_memory=False)
            frame.index = pd.to_datetime(frame.iloc[:, 0])
            for col, key in columns.items():
                if col in frame:
                    series[key].setdefault(turbine.group(1), []).append(
                        pd.to_numeric(frame[col], errors="coerce"))
    for key, turbines in series.items():
        rows = [regular(pd.concat(parts).to_frame("v"), "10min")["v"].to_numpy()
                for parts in turbines.values()]
        emit(f"{subset_prefix}_{key}", rows, "10T", "wind", "CC-BY-4.0", origin)


def wind():
    """Kelmarsh and Penmanshiel turbines, 10-minute means of four physical channels."""
    if not done("kelmarsh_power"):
        files = {f: f"https://zenodo.org/api/records/16807551/files/{f}/content"
                 for f in ("Kelmarsh_SCADA_2018_3084.zip", "Kelmarsh_SCADA_2019_3085.zip",
                           "Kelmarsh_SCADA_2020_3086.zip")}
        _scada([cached(u, f) for f, u in files.items()], "kelmarsh",
               "https://zenodo.org/records/16807551")
    if not done("penmanshiel_power"):
        files = {f: f"https://zenodo.org/api/records/16807304/files/{f}/content"
                 for f in ("Penmanshiel_SCADA_2018_WT11-15_3116.zip",
                           "Penmanshiel_SCADA_2019_WT11-15_3117.zip",
                           "Penmanshiel_SCADA_2020_WT11-15_3118.zip")}
        _scada([cached(u, f) for f, u in files.items()], "penmanshiel",
               "https://zenodo.org/records/16807304")


def pvdaq():
    """Inverters and weather stations of three PV systems, every numeric channel."""
    base = "https://oedi-data-lake.s3.amazonaws.com"
    groups = {}
    for system in ("2105", "2107", "9068"):
        prefix = f"pvdaq/2023-solar-data-prize/{system}_OEDI/data/"
        listing = get(f"{base}/?list-type=2&prefix={prefix}").decode()
        for key, size in zip(re.findall(r"<Key>([^<]+\.csv)</Key>", listing),
                             re.findall(r"<Key>[^<]+\.csv</Key>.*?<Size>(\d+)", listing)):
            if int(size) > 400_000_000:
                continue
            path = cached(f"{base}/{key}", f"pvdaq/{key.rsplit('/', 1)[-1]}")
            frame = pd.read_csv(path, low_memory=False)
            time_col = next(c for c in frame if "measured_on" in c or "time" in c.lower())
            frame.index = pd.to_datetime(frame.pop(time_col), errors="coerce")
            frame = frame[frame.index.notna()].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if len(frame) < MIN_OBS:
                continue
            minutes = round(pd.Series(frame.index).sort_values().diff().median().total_seconds() / 60)
            if minutes not in (1, 5, 15):
                continue
            grid = regular(frame, f"{minutes}min")
            groups.setdefault(minutes, []).extend(grid[c].to_numpy() for c in grid)
    for minutes, rows in groups.items():
        emit(f"pvdaq_{minutes}T", rows, f"{minutes}T", "energy", "CC-BY-4.0",
             "https://data.openei.org/submissions/4568")


def _period(date: str, f: str) -> int:
    if f == "A":
        return int(date[:4])
    if f == "M":
        return int(date[:4]) * 12 + int(date[4:6])
    if f == "Q":
        return int(date[:4]) * 4 + int(date[-1])
    return int(pd.Timestamp(date).toordinal() // {"W": 7, "D": 1}[f])


def eia():
    """EIA bulk files: every weekly, monthly, quarterly and yearly series."""
    rows = {}
    for name in ("PET", "NG", "TOTAL", "SEDS", "COAL", "AEO"):
        try:
            z = zipfile.ZipFile(cached(f"https://api.eia.gov/bulk/{name}.zip"))
        except Exception as e:
            print(name, e, flush=True)
            continue
        with z.open(z.namelist()[0]) as f:
            for line in f:
                item = json.loads(line)
                freq, data = item.get("f"), item.get("data")
                if freq not in ("W", "M", "Q", "A") or not data:
                    continue
                steps = np.array([_period(str(d), freq) for d, _ in data])
                x = np.full(steps.max() - steps.min() + 1, np.nan, dtype=np.float32)
                for s, (_, v) in zip(steps, data):
                    if isinstance(v, (int, float)):
                        x[s - steps.min()] = v
                rows.setdefault(freq, []).append(x)
    names = {"W": "W-SUN", "M": "MS", "Q": "Q-DEC", "A": "A-DEC"}
    for f, series in rows.items():
        emit(f"eia_{f}", series, names[f], "energy", "Public domain (US EIA)",
             "https://www.eia.gov/opendata/")


def wdi():
    """World Development Indicators, one yearly series per country and indicator."""
    if done("wdi_A"):
        return
    z = zipfile.ZipFile(cached("https://databank.worldbank.org/data/download/WDI_CSV.zip"))
    name = next(n for n in z.namelist() if re.fullmatch(r"WDI\w*(Data|CSV)\.csv", n))
    with z.open(name) as f:
        frame = pd.read_csv(f, low_memory=False)
    years = [c for c in frame if re.fullmatch(r"\d{4}", c)]
    values = frame[years].to_numpy(np.float32)
    keep = np.isfinite(values).sum(1) >= 20
    values = values[keep]
    rng = np.random.default_rng(0)
    values = values[rng.permutation(len(values))[:60000]]
    emit("wdi_A", list(values), "A-DEC", "economics", "CC-BY-4.0",
         "https://datacatalog.worldbank.org/search/dataset/0037712")


def _socrata(dataset: str) -> list:
    rows, offset = [], 0
    while True:
        url = (f"https://data.cdc.gov/resource/{dataset}.json?$limit=50000&$offset={offset}"
               "&$order=:id")
        page = json.loads(get(url))
        rows += page
        offset += len(page)
        if len(page) < 50000:
            return rows


def cdc():
    """Weekly deaths by state and cause, 2014-2019 and 2020-2023, then notifiable diseases."""
    if not done("cdc_deaths_W"):
        series = []
        for dataset in ("3yf8-kanr", "muzy-jte6"):
            frame = pd.DataFrame(_socrata(dataset))
            key = next(c for c in frame if "jurisdiction" in c)
            week = next(c for c in frame if "week" in c and "end" in c)
            frame[week] = pd.to_datetime(frame[week])
            causes = [c for c in frame if c not in (key, week, "data_as_of") and not c.startswith("flag")
                      and pd.to_numeric(frame[c], errors="coerce").notna().mean() > 0.5
                      and c not in ("mmwryear", "mmwrweek")]
            for _, g in frame.groupby(key):
                grid = regular(g.set_index(week)[causes].apply(pd.to_numeric, errors="coerce"), "7D")
                series.extend(grid[c].to_numpy() for c in causes)
        emit("cdc_deaths_W", series, "W-SUN", "health", "Public domain (US CDC)",
             "https://data.cdc.gov/d/muzy-jte6")
    if not done("cdc_nndss_W"):
        frame = pd.DataFrame(_socrata("x9gk-5huc"))
        frame["step"] = pd.to_numeric(frame["year"]) * 52 + pd.to_numeric(frame["week"])
        frame["v"] = pd.to_numeric(frame["m1"], errors="coerce")
        series = []
        for _, g in frame.groupby(["states", "label"]):
            x = np.full(int(g["step"].max() - g["step"].min()) + 1, np.nan)
            x[(g["step"] - g["step"].min()).astype(int).to_numpy()] = g["v"].to_numpy()
            series.append(x)
        emit("cdc_nndss_W", series, "W-SUN", "health", "Public domain (US CDC)",
             "https://data.cdc.gov/d/x9gk-5huc")


REST = "https://wikimedia.org/api/rest_v1/metrics/pageviews"


def _article(args):
    project, title = args
    quoted = urllib.parse.quote(title, safe="")
    try:
        items = json.loads(get(f"{REST}/per-article/{project}/all-access/user/{quoted}"
                               "/daily/2016010100/2025093000"))["items"]
    except Exception:
        return None
    s = pd.Series([i["views"] for i in items],
                  index=pd.to_datetime([i["timestamp"][:8] for i in items]), dtype=float)
    return s.resample("W-SUN").sum(min_count=7).iloc[1:-1].to_numpy()


def wikipedia():
    """Project totals (hourly) and weekly views of articles from several monthly top lists."""
    projects = ["en", "de", "fr", "es", "ja", "ru", "it", "pt", "zh", "pl", "nl", "ar", "ko", "tr",
                "sv", "fa", "id", "vi", "uk", "he"]
    if not done("wikipedia_projects_H"):
        rows = []
        for p in projects:
            for project in (f"{p}.wikipedia", f"{p}.wiktionary"):
                try:
                    items = json.loads(get(f"{REST}/aggregate/{project}/all-access/user/hourly/"
                                           "2016010100/2025093000"))["items"]
                except Exception:
                    continue
                s = pd.Series([i["views"] for i in items], dtype=float,
                              index=pd.to_datetime([i["timestamp"] for i in items], format="%Y%m%d%H"))
                rows.append(regular(s.to_frame("v"), "h")["v"].to_numpy())
        emit("wikipedia_projects_H", rows, "H", "web", "CC0-1.0",
             "https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews")
    if not done("wikipedia_articles_W"):
        titles = set()
        for p in projects:
            for year, month in ((2017, "06"), (2019, "03"), (2021, "09"), (2023, "12"), (2025, "06")):
                try:
                    top = json.loads(get(f"{REST}/top/{p}.wikipedia/all-access/{year}/{month}/all-days"))
                except Exception:
                    continue
                titles |= {(f"{p}.wikipedia", a["article"]) for a in top["items"][0]["articles"]}
        titles = sorted(t for t in titles if ":" not in t[1])
        rng = np.random.default_rng(0)
        picks = [titles[i] for i in rng.permutation(len(titles))[:12000]]
        with cf.ThreadPoolExecutor(8) as pool:
            rows = [r for r in pool.map(_article, picks) if r is not None]
        emit("wikipedia_articles_W", rows, "W-SUN", "web", "CC0-1.0",
             "https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews")


def _site(args):
    site, years = args
    path = RAW / "webtris" / f"{site}.npy"
    if path.exists():
        return np.load(path)
    parts = []
    for year in years:
        start, end = f"0101{year}", f"3112{year}" if year < 2025 else "30092025"
        try:
            page = json.loads(get("https://webtris.nationalhighways.co.uk/api/v1/reports/Daily?"
                                  f"sites={site}&start_date={start}&end_date={end}"
                                  "&page=1&page_size=40000"))
        except Exception:
            continue
        for row in page.get("Rows", []):
            value = row.get("Total Volume")
            if value not in (None, ""):
                t = pd.Timestamp(row["Report Date"]) + pd.Timedelta(minutes=15 * int(row["Time Interval"]))
                parts.append((t, float(value)))
    if len(parts) < MIN_OBS:
        return None
    s = pd.Series(dict(parts))
    x = regular(s.to_frame("v"), "15min")["v"].to_numpy()
    path.parent.mkdir(exist_ok=True)
    np.save(path, x)
    return x


def webtris():
    """Total flow at a sample of English motorway sites, 15 minutes, 2024 to 2025-09."""
    if done("webtris_15T"):
        return
    sites = [s["Id"] for s in json.loads(get("https://webtris.nationalhighways.co.uk/api/v1/sites"))["sites"]
             if s["Status"] == "Active"]
    rng = np.random.default_rng(0)
    picks = [sites[i] for i in rng.permutation(len(sites))[:400]]
    with cf.ThreadPoolExecutor(10) as pool:
        rows = [r for r in pool.map(_site, [(s, (2024, 2025)) for s in picks]) if r is not None]
    emit("webtris_15T", rows, "15T", "road traffic", "OGL-UK-3.0",
         "https://webtris.nationalhighways.co.uk/api/swagger/ui/index")


def _institution_csvs(tar_path: Path, folder: str):
    with tarfile.open(tar_path) as t:
        for member in t:
            if member.isfile() and folder in member.name and member.name.endswith(".csv"):
                yield pd.read_csv(t.extractfile(member))


def cesnet():
    """Network traffic of institutions, 10-minute and hourly aggregates."""
    tar = cached("https://zenodo.org/api/records/13382427/files/institutions.tar.gz/content",
                 "cesnet_institutions.tar.gz")
    metrics = ("n_flows", "n_packets", "n_bytes", "n_dest_ip", "n_dest_ports", "avg_duration")
    for folder, freq, subset in (("agg_10_minutes", "10T", "cesnet_10T"),
                                 ("agg_1_hour", "H", "cesnet_H")):
        if done(subset):
            continue
        rows = []
        for frame in _institution_csvs(tar, folder):
            frame = frame[frame["id_time"].notna()]
            steps = frame["id_time"].to_numpy(np.int64)
            for m in metrics:
                if m in frame and len(steps):
                    x = np.full(steps.max() + 1, np.nan)
                    x[steps] = frame[m].to_numpy()
                    rows.append(x)
        emit(subset, rows, freq, "cloud computing", "CC-BY-4.0", "https://zenodo.org/records/13382427")


def _ten_seconds(frame: pd.DataFrame, subset_rows: list) -> None:
    grid = frame.resample("10s").mean()
    subset_rows.extend(grid[c].to_numpy() for c in grid)


def darck():
    """Mains and appliances of one apartment, 1 Hz reduced to 10-second means."""
    if done("darck_10S"):
        return
    path = cached("https://zenodo.org/api/records/17159850/files/DARCK.csv.xz/content", "DARCK.csv.xz")
    frame = pd.read_csv(path, compression="xz", low_memory=False)
    index = pd.to_datetime(frame.pop("time"), errors="coerce", utc=True, format="ISO8601")
    frame.index = pd.DatetimeIndex(index).tz_localize(None)
    rows = []
    _ten_seconds(frame[frame.index.notna()].apply(pd.to_numeric, errors="coerce"), rows)
    emit("darck_10S", rows, "10S", "energy", "CC-BY-4.0", "https://zenodo.org/records/17159850")


def rae():
    """Mains and sub-meters of two houses, 1 Hz reduced to 10-second means."""
    if done("rae_10S"):
        return
    rows = []
    for name, fid in (("house1_power_blk1", 3094551), ("house1_power_blk2", 3094552),
                      ("house2_power_blk1", 3094559)):
        path = cached(f"https://dataverse.harvard.edu/api/access/datafile/{fid}?format=original",
                      f"rae_{name}.csv")
        frame = pd.read_csv(path)
        frame.index = pd.to_datetime(frame.pop("unix_ts"), unit="s")
        _ten_seconds(frame.drop(columns="ihd", errors="ignore"), rows)
    emit("rae_10S", rows, "10S", "energy", "CC-BY-4.0", "https://doi.org/10.7910/DVN/ZJW4LC")


def n2o():
    """Wastewater treatment plant signals, values with a non-zero quality flag removed."""
    if done("n2o_wastewater"):
        return
    url = ("https://data.mendeley.com/public-files/datasets/xmbxhscgpr/files/"
           "6b064aee-f2f2-47f7-8078-4fae838a2a88/file_downloaded")
    frame = pd.read_csv(cached(url, "aved_raw.csv"), low_memory=False)
    index = pd.to_datetime(frame.pop("time"), errors="coerce", utc=True, format="ISO8601")
    frame.index = pd.DatetimeIndex(index).tz_localize(None)
    frame = frame[frame.index.notna()]
    values = {}
    for col in frame:
        if col.endswith(" value"):
            v = pd.to_numeric(frame[col], errors="coerce")
            flag = pd.to_numeric(frame[col.replace(" value", " quality")], errors="coerce")
            values[col] = v.where(flag == 0)
    grid = pd.DataFrame(values)
    minutes = max(1, round(pd.Series(grid.index).sort_values().diff().median().total_seconds() / 60))
    grid = grid.resample(f"{minutes}min").mean()
    emit("n2o_wastewater", [grid[c].to_numpy() for c in grid], f"{minutes}T", "industry",
         "CC-BY-4.0", "https://doi.org/10.17632/xmbxhscgpr.4")


def nsw():
    """Permanent traffic counters of New South Wales, hourly, one series per direction."""
    if done("nsw_traffic_H"):
        return
    meta = json.loads(get("https://opendata.transport.nsw.gov.au/api/3/action/package_show"
                          "?id=nsw-roads-traffic-volume-counts-api"))
    url = next(r["url"] for r in meta["result"]["resources"] if "Hourly Permanent" in r["name"])
    z = zipfile.ZipFile(cached(url, "nsw_hourly_permanent.zip"))
    hours = [f"hour_{h:02d}" for h in range(24)]
    keys = ["station_key", "traffic_direction_seq", "cardinal_direction_seq", "classification_seq"]
    parts = []
    for name in z.namelist():
        if name.endswith(".csv"):
            with z.open(name) as f:
                parts.append(pd.read_csv(f, usecols=keys + ["date"] + hours, low_memory=False))
    df = pd.concat(parts)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    df = df[df["date"].notna() & (df["date"] < "2025-10-01")]
    rows = []
    for _, g in df.groupby(keys, sort=False):
        g = g.drop_duplicates("date").sort_values("date").set_index("date")[hours]
        grid = g.reindex(pd.date_range(g.index[0], g.index[-1], freq="D")).to_numpy(np.float32)
        rows.append(grid.ravel())
    emit("nsw_traffic_H", rows, "H", "road traffic", "CC-BY-4.0",
         "https://opendata.transport.nsw.gov.au/dataset/nsw-roads-traffic-volume-counts-api")


def bosch():
    """Vibration of CNC machines: 2 kHz reduced to a 1-second RMS, runs of one operation joined."""
    if done("bosch_cnc_1S"):
        return
    import subprocess

    import h5py

    repo = RAW / "CNC_Machining"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/boschresearch/CNC_Machining", str(repo)], check=True)
    groups = {}
    for path in sorted(repo.glob("data/*/*/*/*.h5")):
        machine, op = path.parts[-4], path.parts[-3]
        with h5py.File(path) as f:
            x = np.asarray(f["vibration_data"], dtype=np.float64)
        n = len(x) // 2000
        if n:
            rms = np.sqrt((x[:n * 2000].reshape(n, 2000, -1) ** 2).mean(1))
            groups.setdefault((machine, op), []).append(rms)
    rows = []
    for runs in groups.values():
        joined = np.concatenate(runs)
        rows.extend(joined[:, i] for i in range(joined.shape[1]))
    emit("bosch_cnc_1S", rows, "1S", "industry", "BSD-3-Clause",
         "https://github.com/boschresearch/CNC_Machining")


def eia930():
    """EIA-930 hourly grid monitor: demand, forecast, generation by fuel and interchange of
    every balancing authority, UTC series only."""
    if done("eia930_H"):
        return
    z = zipfile.ZipFile(cached("https://www.eia.gov/opendata/bulk/EBA.zip"))
    rows = []
    with z.open(z.namelist()[0]) as f:
        for line in f:
            item = json.loads(line)
            data = item.get("data")
            if item.get("f") != "H" or not data:
                continue
            # Times read "20150701T05Z" in older files, "20150701T05" in newer ones, UTC both.
            stamps = [str(d).rstrip("Z") for d, _ in data]
            hours = pd.to_datetime(stamps, format="%Y%m%dT%H").asi8 // 3_600_000_000_000
            x = np.full(hours.max() - hours.min() + 1, np.nan, dtype=np.float32)
            # Values come as strings in the current files, numbers in older ones.
            values = pd.to_numeric(pd.Series([v for _, v in data]), errors="coerce")
            values = values.to_numpy(np.float32)
            x[hours - hours.min()] = values
            rows.append(x)
    emit("eia930_H", rows, "H", "energy", "Public domain (US EIA)",
         "https://www.eia.gov/electricity/gridmonitor/")


def google_power():
    """Measured power of the 57 power domains of Google's 2019 cluster trace, 5 minutes,
    points flagged as bad measurements removed."""
    if done("google_power_5T"):
        return
    base = "https://storage.googleapis.com/powerdata_2019/"
    listing = json.loads(get("https://storage.googleapis.com/storage/v1/b/powerdata_2019/o"))
    rows = []
    for item in listing["items"]:
        name = item["name"]
        if not name.startswith("cell"):
            continue
        df = pd.read_csv(cached(base + name, f"powerdata_2019/{name}"))
        df = df[~df.bad_measurement_data.astype(str).str.lower().eq("true")]
        step = (df.time.to_numpy(np.int64) - df.time.min()) // 300_000_000
        x = np.full(step.max() + 1, np.nan, dtype=np.float32)
        x[step] = df.measured_power_util.to_numpy(np.float32)
        rows.append(x)
    emit("google_power_5T", rows, "5T", "energy", "CC-BY-4.0",
         "https://github.com/google/cluster-data/blob/master/PowerData2019.md")


def azure_functions_2021():
    """Invocations per minute of each function of the 2021 trace, counted by end minute,
    functions with fewer than 200 invocations in the two weeks left out."""
    if done("azure_functions_2021"):
        return
    import subprocess

    archive = cached("https://github.com/Azure/AzurePublicDataset/raw/master/data/"
                     "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar")
    folder = RAW / "azure_functions_2021"
    if not any(p.stat().st_size for p in folder.rglob("*.txt")):
        # RAR5 with a large dictionary: needs unrar (libarchive and the free 7-Zip refuse it).
        folder.mkdir(exist_ok=True)
        subprocess.run(["unrar", "x", "-o+", str(archive), f"{folder}/"], check=True,
                       stdout=subprocess.DEVNULL)
    df = pd.concat(pd.read_csv(p) for p in folder.rglob("*.txt") if p.stat().st_size)
    minute = (df.end_timestamp // 60).astype(np.int64)
    span = int(minute.max()) + 1
    rows = []
    for _, m in minute.groupby([df.app, df.func]):
        if len(m) >= 200:
            rows.append(np.bincount(m.to_numpy(), minlength=span).astype(np.float32))
    emit("azure_functions_2021", rows, "T", "Web/CloudOps", "CC-BY-4.0",
         "https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md")


FETCHERS = {f.__name__: f for f in (epa, wind, pvdaq, eia, wdi, cdc, wikipedia, webtris, cesnet,
                                    darck, rae, n2o, nsw, bosch, eia930, google_power,
                                    azure_functions_2021)}


def main():
    global RAW, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", default=list(FETCHERS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    a = ap.parse_args()
    RAW, OUT = a.raw, a.out
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.sources:
        try:
            FETCHERS[name]()
        except Exception as e:
            print(f"{name} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
