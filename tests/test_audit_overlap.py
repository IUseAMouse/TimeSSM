import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = pytest.importorskip("datasets") and load("audit_overlap")
merge = load("merge_corpus")


def walk(n, seed):
    return np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.05, n)))


def test_rescaled_stretch_shares_every_window_at_one_offset():
    x = walk(500, 0)
    h1, p1 = audit.hashes(x, positions=True)
    h2, p2 = audit.hashes(3.7 * x[100:400], positions=True)
    common, i1, i2 = np.intersect1d(h1, h2, return_indices=True)
    assert len(common) == len(h2)
    assert set(p1[i1] - p2[i2]) == {100}


def test_independent_walks_share_nothing():
    assert len(np.intersect1d(audit.hashes(walk(2000, 1)), audit.hashes(walk(2000, 2)))) == 0


def test_flat_and_constant_growth_windows_are_skipped():
    assert len(audit.hashes(np.ones(100))) == 0
    assert len(audit.hashes(1.01 ** np.arange(100))) == 0


def test_merge_drops_listed_series_and_keeps_the_source(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    series = [np.arange(n, dtype=np.float32) + 10 * n for n in (5, 7, 3, 4)]
    values = np.concatenate(series)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in series])]).astype(np.int64)
    np.save(src / "a.values.npy", values)
    np.save(src / "a.offsets.npy", offsets)
    np.save(src / "b.values.npy", values)
    np.save(src / "b.offsets.npy", offsets)
    row = {"freq": "D", "n_series": 4, "n_obs": int(len(values))}
    (src / "manifest.json").write_text(json.dumps({"a": row, "b": row}))
    report = tmp_path / "leaks.json"
    report.write_text(json.dumps([{"subset": "a", "leaking_series": [1, 3]}]))
    sys.argv = ["merge_corpus", "--out", str(out), "--drop", str(report), str(src)]
    merge.main()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["a"]["n_series"] == 2 and manifest["a"]["n_obs"] == 8
    v, o = np.load(out / "a.values.npy"), np.load(out / "a.offsets.npy")
    np.testing.assert_array_equal(v[o[0]:o[1]], series[0])
    np.testing.assert_array_equal(v[o[1]:o[2]], series[2])
    assert (out / "b.values.npy").is_symlink()
    np.testing.assert_array_equal(np.load(src / "a.values.npy"), values)
    with pytest.raises(SystemExit):
        merge.main()
