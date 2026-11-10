"""
External sources of corpus v4, one fetcher per source (docs/DATA.md lists them).

    python scripts/fetch_sources.py --raw /workspace/data/ext_raw \
        --out /workspace/data/v4_ext azure_functions uci_household

Same layout as scripts/build_corpus_v4.py: <subset>.values.npy (float32, series
end to end, NaN = missing), <subset>.offsets.npy, and manifest.json, which is
updated in place. Downloads are cached under --raw.
"""

import argparse
import gzip
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import pandas as pd

MIN_LENGTH = 16
SEED = 0
USER_AGENT = {"User-Agent": "timessm-corpus/1.0"}
# WDI indicators behind fev-bench tasks (CO2 emissions, life expectancy, tourism).
FEV_INDICATORS = ("EN.ATM.CO2", "EN.GHG.CO2", "SP.DYN.LE00", "ST.INT")


def get(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=600) as r:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(path)
    return path


def text(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=120) as r:
        return r.read().decode()


def write(out: Path, name: str, series: Iterable[np.ndarray], meta: dict) -> dict:
    """Strip leading and trailing NaN, drop what is too short, save, update the manifest."""
    kept: List[np.ndarray] = []
    for x in series:
        x = np.asarray(x, dtype=np.float32)
        obs = np.flatnonzero(np.isfinite(x))
        if len(obs) >= MIN_LENGTH and np.nanstd(x) > 0:
            kept.append(x[obs[0]:obs[-1] + 1])
    lengths = [len(x) for x in kept]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(out / f"{name}.values.npy", np.concatenate(kept))
    np.save(out / f"{name}.offsets.npy", offsets)
    row = {"subset": name, "n_series": len(kept), "n_obs": int(offsets[-1]),
           "median_length": int(np.median(lengths)), **meta}
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[name] = row
    path.write_text(json.dumps(manifest, indent=1))
    print(row, flush=True)
    return row


def on_grid(ordinals: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Integer time ordinals -> regular series, NaN in the gaps."""
    x = np.full(int(ordinals.max() - ordinals.min()) + 1, np.nan, dtype=np.float32)
    x[ordinals - ordinals.min()] = values
    return x


# ------------------------------------------------------------ cloud and operations
def azure_functions(raw: Path, out: Path) -> None:
    """Invocations per minute of 4000 sampled functions over the 14 days of the trace."""
    url = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/"
           "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz")
    archive = get(url, raw / "azure_functions_2019.tar.xz")
    days = raw / "azure_functions_2019"
    if not days.exists():
        days.mkdir()
        subprocess.run(["tar", "xJf", str(archive), "-C", str(days), "--wildcards",
                        "invocations_per_function_md.anon.d*.csv"], check=True)
    minutes = [str(i) for i in range(1, 1441)]
    series: dict = {}
    for path in sorted(days.glob("invocations_per_function_md.anon.d*.csv")):
        day = int(re.search(r"\.d(\d+)\.csv", path.name).group(1)) - 1
        df = pd.read_csv(path, engine="pyarrow")
        # A fixed hash bucket picks the same functions every day (about 4000 of 72000).
        chosen = df[df.HashFunction.str[:2].map(lambda h: int(h, 16) % 18 == 0)]
        for f, row in zip(chosen.HashFunction, chosen[minutes].to_numpy(np.float32)):
            series.setdefault(f, np.full((14, 1440), np.nan, np.float32))[day] = row
    write(out, "azure_functions_2019", (x.ravel() for x in series.values()),
          {"freq": "T", "domain": "Web/CloudOps", "license": "CC-BY-4.0", "origin": url})


def exathlon(raw: Path, out: Path) -> None:
    """Spark cluster metrics at 1 s: every numeric channel with variance of the sampled traces."""
    repo = raw / "exathlon"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/exathlonbenchmark/exathlon", str(repo)], check=True)
    rng = np.random.default_rng(SEED)
    series: List[np.ndarray] = []
    for archive in sorted(repo.glob("data/raw/app*/*.zip")):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.endswith(".csv") or z.getinfo(name).file_size < 100_000:
                    continue
                df = pd.read_csv(z.open(name), low_memory=False).select_dtypes("number")
                columns = [c for c in df.columns if df[c].nunique() > 16]
                for c in rng.choice(columns, min(24, len(columns)), replace=False):
                    series.append(df[c].to_numpy(np.float32))
    write(out, "exathlon_spark", series,
          {"freq": "1S", "domain": "Web/CloudOps", "license": "Apache-2.0",
           "origin": "https://github.com/exathlonbenchmark/exathlon"})


def rcaeval(raw: Path, out: Path) -> None:
    """Microservice metrics at 1 s from three testbeds, with injected faults."""
    base = "https://zenodo.org/api/records/14590730/files"
    rng = np.random.default_rng(SEED)
    series: List[np.ndarray] = []
    for name in ("RE1-OB", "RE1-SS", "RE1-TT", "RE2-OB", "RE2-SS"):
        path = get(f"{base}/{name}.zip/content", raw / "rcaeval" / f"{name}.zip")
        with zipfile.ZipFile(path) as z:
            cases = sorted(n for n in z.namelist() if n.endswith(("data.csv", "metrics.csv")))
            for case in rng.choice(cases, min(150, len(cases)), replace=False):
                df = pd.read_csv(z.open(str(case))).select_dtypes("number")
                columns = [c for c in df.columns if c != "time" and df[c].nunique() > 16]
                for c in rng.choice(columns, min(24, len(columns)), replace=False):
                    series.append(df[c].to_numpy(np.float32))
    write(out, "rcaeval_microservices", series,
          {"freq": "1S", "domain": "Web/CloudOps", "license": "CC-BY-4.0",
           "origin": "https://doi.org/10.5281/zenodo.14590730"})


def smd(raw: Path, out: Path) -> None:
    """Server Machine Dataset: 38 metrics per minute of 28 machines, train and test joined."""
    repo = raw / "omnianomaly"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/NetManAIOps/OmniAnomaly", str(repo)], check=True)
    series: List[np.ndarray] = []
    for train in sorted(repo.glob("ServerMachineDataset/train/*.txt")):
        test = train.parent.parent / "test" / train.name
        table = np.concatenate([np.loadtxt(train, delimiter=","), np.loadtxt(test, delimiter=",")])
        series.extend(table.T)
    write(out, "smd_servers", series,
          {"freq": "T", "domain": "Web/CloudOps", "license": "MIT",
           "origin": "https://github.com/NetManAIOps/OmniAnomaly"})


# ------------------------------------------------------------ energy and industry
def uci_hydraulic(raw: Path, out: Path) -> None:
    """Hydraulic test rig: 60 s cycles concatenated, 100 Hz and 10 Hz channels averaged to 1 Hz."""
    url = ("https://archive.ics.uci.edu/static/public/447/"
           "condition+monitoring+of+hydraulic+systems.zip")
    series = []
    with zipfile.ZipFile(get(url, raw / "uci447.zip")) as z:
        for name in sorted(z.namelist()):
            if re.fullmatch(r"(PS\d|EPS1|FS\d|TS\d|VS1|CE|CP|SE)\.txt", name):
                table = np.loadtxt(z.open(name), dtype=np.float32)
                per_second = table.shape[1] // 60
                series.append(table.reshape(len(table), 60, per_second).mean(-1).ravel())
    write(out, "uci_hydraulic", series,
          {"freq": "1S", "domain": "Industry", "license": "CC-BY-4.0", "origin": url})


def uci_household(raw: Path, out: Path) -> None:
    url = ("https://archive.ics.uci.edu/static/public/235/"
           "individual+household+electric+power+consumption.zip")
    with zipfile.ZipFile(get(url, raw / "uci235.zip")) as z:
        df = pd.read_csv(z.open("household_power_consumption.txt"), sep=";", na_values="?",
                         low_memory=False)
    write(out, "uci_household_power", (df[c].to_numpy() for c in df.columns[2:]),
          {"freq": "T", "domain": "Energy", "license": "CC-BY-4.0", "origin": url})


def uci_steel(raw: Path, out: Path) -> None:
    url = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
    with zipfile.ZipFile(get(url, raw / "uci851.zip")) as z:
        df = pd.read_csv(z.open("Steel_industry_data.csv"))
    write(out, "uci_steel", (df[c].to_numpy() for c in df.select_dtypes("number").columns
                             if c != "NSM"),
          {"freq": "15T", "domain": "Industry", "license": "CC-BY-4.0", "origin": url})


def borealis_6s(raw: Path, out: Path) -> None:
    """Six-second household load, one series per home and measured channel."""
    doi = "doi:10.5683/SP2/R4SVBF"
    base = "https://borealisdata.ca/api"
    info = json.loads(text(f"{base}/datasets/:persistentId/?persistentId={doi}"))
    series = []
    for f in info["data"]["latestVersion"]["files"]:
        name = f["dataFile"]["filename"]
        if not name.endswith(".tab"):
            continue
        path = get(f"{base}/access/datafile/{f['dataFile']['id']}?format=original",
                   raw / "borealis" / name)
        df = pd.read_csv(path, sep=None, engine="python", nrows=5)
        numeric = df.select_dtypes("number").columns
        df = pd.read_csv(path, sep=None, engine="python", usecols=numeric, dtype=np.float32)
        series.extend(df[c].to_numpy() for c in df.columns[-2:])
    write(out, "borealis_6s", series,
          {"freq": "6S", "domain": "Energy", "license": "CC0-1.0",
           "origin": "https://doi.org/10.5683/SP2/R4SVBF"})


# ------------------------------------------------------------ official statistics
def eurostat_series(raw: Path, table: str, freq: str, keep: dict = None) -> Iterator[np.ndarray]:
    url = ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
           f"{table}?format=SDMX-CSV&compressed=true")
    path = get(url, raw / "eurostat" / f"{table}.csv.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, low_memory=False)
    for column, value in (keep or {}).items():
        df = df[df[column].isin(value)] if column in df else df
    drop = {"DATAFLOW", "LAST UPDATE", "freq", "TIME_PERIOD", "OBS_VALUE", "OBS_FLAG",
            "OBS_STATUS", "CONF_STATUS"}
    key = [c for c in df.columns if c not in drop]
    df = df.dropna(subset=["OBS_VALUE"])
    df = df.assign(t=pd.PeriodIndex(df.TIME_PERIOD.astype(str), freq=freq).astype("int64"))
    for _, group in df.groupby(key, sort=False):
        yield on_grid(group.t.to_numpy(), group.OBS_VALUE.to_numpy(np.float32))


def eurostat_monthly(raw: Path, out: Path) -> None:
    series: List[np.ndarray] = []
    for table in ("sts_inpr_m", "sts_trtu_m", "sts_copr_m"):
        series.extend(eurostat_series(raw, table, "M", {"s_adj": ["NSA"]}))
    write(out, "eurostat_sts_m", series,
          {"freq": "M", "domain": "Econ/Fin", "license": "Eurostat re-use policy (attribution)",
           "origin": "https://ec.europa.eu/eurostat/web/short-term-business-statistics"})


def eurostat_national(raw: Path, out: Path) -> None:
    write(out, "eurostat_gdp_q",
          eurostat_series(raw, "namq_10_gdp", "Q", {"s_adj": ["NSA"], "unit": ["CP_MEUR"]}),
          {"freq": "Q", "domain": "Econ/Fin", "license": "Eurostat re-use policy (attribution)",
           "origin": "https://ec.europa.eu/eurostat/web/national-accounts"})
    write(out, "eurostat_gdp_a", eurostat_series(raw, "nama_10_gdp", "Y", {"unit": ["CP_MEUR"]}),
          {"freq": "A", "domain": "Econ/Fin", "license": "Eurostat re-use policy (attribution)",
           "origin": "https://ec.europa.eu/eurostat/web/national-accounts"})


def statcan(raw: Path, out: Path) -> None:
    series: List[np.ndarray] = []
    for table in ("16100047", "20100008", "25100015"):
        url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{table}-eng.zip"
        with zipfile.ZipFile(get(url, raw / "statcan" / f"{table}.zip")) as z:
            df = pd.read_csv(z.open(f"{table}.csv"), usecols=["REF_DATE", "VECTOR", "VALUE"],
                             low_memory=False)
        df = df.dropna(subset=["VALUE"])
        df = df.assign(t=pd.PeriodIndex(df.REF_DATE.astype(str), freq="M").astype("int64"))
        series.extend(on_grid(g.t.to_numpy(), g.VALUE.to_numpy(np.float32))
                      for _, g in df.groupby("VECTOR", sort=False))
    write(out, "statcan_m", series,
          {"freq": "M", "domain": "Econ/Fin", "license": "Statistics Canada Open Licence",
           "origin": "https://www150.statcan.gc.ca/n1/tbl/csv/"})


def worldbank_wdi(raw: Path, out: Path) -> None:
    """World Development Indicators, annual, per country and indicator."""
    url = "https://databank.worldbank.org/data/download/WDI_CSV.zip"
    with zipfile.ZipFile(get(url, raw / "wdi.zip")) as z:
        name = next(n for n in z.namelist() if re.search(r"WDI\w*CSV\.csv$", n) or
                    n == "WDIData.csv")
        df = pd.read_csv(z.open(name))
    df = df[~df["Indicator Code"].str.startswith(FEV_INDICATORS)]
    years = [c for c in df.columns if re.fullmatch(r"\d{4}", c)]
    write(out, "worldbank_wdi_a", df[years].to_numpy(np.float32),
          {"freq": "A", "domain": "Econ/Fin", "license": "CC-BY-4.0", "origin": url})


def eurostat_weekly(raw: Path, out: Path) -> None:
    """Weekly deaths by region, age and sex."""
    url = ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
           "demo_r_mwk_ts?format=SDMX-CSV&compressed=true")
    path = get(url, raw / "eurostat" / "demo_r_mwk_ts.csv.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, low_memory=False).dropna(subset=["OBS_VALUE"])
    year, week = df.TIME_PERIOD.str[:4].astype(int), df.TIME_PERIOD.str[-2:].astype(int)
    df = df.assign(t=(year * 53 + week).to_numpy())
    key = [c for c in df.columns if c in ("sex", "unit", "geo", "age")]
    series = [on_grid(g.t.to_numpy(), g.OBS_VALUE.to_numpy(np.float32))
              for _, g in df.groupby(key, sort=False)]
    write(out, "eurostat_deaths_w", series,
          {"freq": "W", "domain": "Healthcare", "license": "Eurostat re-use policy (attribution)",
           "origin": url})


# ------------------------------------------------------------ environment
def noaa_water(raw: Path, out: Path) -> None:
    """Verified hourly water levels of 120 stations, 2015-2024."""
    api = "https://api.tidesandcurrents.noaa.gov"
    stations = json.loads(text(f"{api}/mdapi/prod/webapi/stations.json?type=waterlevels"))
    ids = [s["id"] for s in stations["stations"]]
    rng = np.random.default_rng(SEED)
    series = []
    for sid in rng.choice(ids, min(120, len(ids)), replace=False):
        years = []
        for year in range(2015, 2025):
            url = (f"{api}/api/prod/datagetter?product=hourly_height&application=timessm"
                   f"&begin_date={year}0101&end_date={year}1231&datum=MSL&station={sid}"
                   "&time_zone=gmt&units=metric&format=csv")
            try:
                path = get(url, raw / "noaa" / f"{sid}_{year}.csv")
                df = pd.read_csv(path, usecols=[0, 1], names=["t", "v"], header=0,
                                 parse_dates=["t"])
                years.append(df.dropna())
            except Exception:
                continue
        if years:
            df = pd.concat(years)
            if len(df) >= MIN_LENGTH:
                hours = (df.t - pd.Timestamp("2015-01-01")).dt.total_seconds().to_numpy() // 3600
                series.append(on_grid(hours.astype(np.int64), df.v.to_numpy(np.float32)))
    write(out, "noaa_water_levels", series,
          {"freq": "H", "domain": "Nature", "license": "Public domain (US Government work)",
           "origin": "https://tidesandcurrents.noaa.gov/"})


def usgs_streamflow(raw: Path, out: Path) -> None:
    """Instantaneous discharge at 15 minutes, 2023-2024, stations of 12 states."""
    base = "https://waterservices.usgs.gov/nwis"
    rng = np.random.default_rng(SEED)
    series = []
    for state in ("co", "wa", "tx", "ny", "ca", "fl", "mt", "pa", "or", "az", "mn", "ga"):
        try:
            sites = [ln.split("\t")[1] for ln in text(
                f"{base}/site/?format=rdb&stateCd={state}&parameterCd=00060&siteType=ST"
                "&siteStatus=active&hasDataTypeCd=iv").splitlines()
                if ln.startswith("USGS\t")]
        except Exception as e:
            print(state, e, flush=True)
            continue
        for site in rng.choice(sites, min(25, len(sites)), replace=False):
            url = (f"{base}/iv/?format=rdb&parameterCd=00060&sites={site}"
                   "&startDT=2023-01-01&endDT=2024-12-31")
            try:
                lines = get(url, raw / "usgs" / f"{site}.rdb").read_text().splitlines()
            except Exception:
                continue
            rows = [ln.split("\t") for ln in lines if ln.startswith("USGS\t")]
            if rows:
                series.append(_usgs_grid([(r[2], r[4]) for r in rows if len(r) >= 5]))
    write(out, "usgs_streamflow", series,
          {"freq": "15T", "domain": "Nature", "license": "Public domain (US Government work)",
           "origin": "https://waterservices.usgs.gov/"})


def _usgs_grid(rows) -> np.ndarray:
    df = pd.DataFrame(rows, columns=["t", "v"])
    t = pd.to_datetime(df.t, errors="coerce")
    v = pd.to_numeric(df.v, errors="coerce")
    ok = t.notna() & v.notna()
    if ok.sum() < MIN_LENGTH:
        return np.empty(0)
    minutes = ((t[ok] - t[ok].min()).dt.total_seconds() // 900).to_numpy(np.int64)
    return on_grid(minutes, v[ok].to_numpy(np.float32))


SOURCES = {f.__name__: f for f in (
    azure_functions, exathlon, rcaeval, smd, uci_hydraulic, uci_household, uci_steel, borealis_6s,
    eurostat_monthly, eurostat_national, eurostat_weekly, statcan, worldbank_wdi,
    noaa_water, usgs_streamflow)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", default=[])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.raw.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.sources or SOURCES:
        print("==", name, flush=True)
        try:
            SOURCES[name](a.raw, a.out)
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
