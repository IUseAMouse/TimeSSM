"""
Corpus v4: GiftEvalPretrain converted to one flat file per subset.

    python scripts/build_corpus_v4.py download --raw /workspace/data/raw
    python scripts/build_corpus_v4.py convert --raw /workspace/data/raw --out /workspace/data/v4

Series keep their full length and their NaNs: <subset>.values.npy (float32,
series end to end) and <subset>.offsets.npy (int64, n_series + 1), plus
manifest.json (frequency, series, observations per subset). Missing values
stay missing, the model reads them through a mask.

The large slices of the climate and building subsets are sampled by shard, so
that the download fits on the disk and each year stays represented.
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

REPO = "Salesforce/GiftEvalPretrain"
REVISION = "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874"
MAX_SHARDS = {"buildings_900k": 12, "borg_cluster_data_2011": 12, "azure_vm_traces_2017": 12}
SLICED = re.compile(r"(era5|cmip6)_\d{4}$")
MIN_LENGTH = 16
MAX_OBS = 400_000_000


def wanted(path: str) -> bool:
    subset, _, name = path.partition("/")
    if not name.endswith(".arrow"):
        return name in ("state.json", "dataset_info.json")
    match = re.search(r"data-(\d+)-of", name)
    if match is None:
        return False
    shard = int(match.group(1))
    if SLICED.match(subset):
        return shard == 0
    return shard < MAX_SHARDS.get(subset, 10 ** 6)


def download(raw: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
             if "/" in f and wanted(f)]
    print(f"{len(files)} files")

    def fetch(f):
        return hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=raw)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as pool:
        for i, _ in enumerate(pool.map(fetch, files)):
            if i % 100 == 0:
                print(i, flush=True)


def batches(path: Path):
    with pa.memory_map(str(path)) as source:
        yield from ipc.open_stream(source)


def convert_subset(args) -> dict:
    subset_dir, out = args
    subset = subset_dir.name
    chunks, lengths, freq, total = [], [], None, 0
    for shard in sorted(subset_dir.glob("*.arrow")):
        for batch in batches(shard):
            if freq is None and "freq" in batch.schema.names:
                freq = batch.column("freq")[0].as_py()
            for target in batch.column("target").to_pylist():
                arr = np.asarray(target, dtype=np.float32)
                for channel in arr.reshape(-1, arr.shape[-1]):
                    obs = np.isfinite(channel)
                    if obs.sum() < MIN_LENGTH:
                        continue
                    keep = np.flatnonzero(obs)
                    channel = channel[keep[0]:keep[-1] + 1]
                    chunks.append(channel)
                    lengths.append(len(channel))
                    total += len(channel)
            if total >= MAX_OBS:
                break
        if total >= MAX_OBS:
            break
    if not chunks:
        return {"subset": subset, "n_series": 0, "n_obs": 0, "freq": freq}
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(out / f"{subset}.values.npy", np.concatenate(chunks))
    np.save(out / f"{subset}.offsets.npy", offsets)
    return {"subset": subset, "n_series": len(lengths), "n_obs": int(total), "freq": freq,
            "median_length": int(np.median(lengths))}


def convert(raw: Path, out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    subsets = sorted(p for p in raw.iterdir() if p.is_dir() and any(p.glob("*.arrow")))
    with ProcessPoolExecutor(workers) as pool:
        rows = []
        for row in pool.map(convert_subset, [(s, out) for s in subsets]):
            print(row, flush=True)
            rows.append(row)
    (out / "manifest.json").write_text(json.dumps({r["subset"]: r for r in rows}, indent=1))
    print(f"{len(rows)} subsets, {sum(r['n_obs'] for r in rows) / 1e9:.2f} B observations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["download", "convert"])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    download(a.raw) if a.command == "download" else convert(a.raw, a.out, a.workers)


if __name__ == "__main__":
    main()
