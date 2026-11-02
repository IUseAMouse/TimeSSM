import json

import numpy as np
import pytest

from timessm.corpus import WindowBatches, train_subsets, DEV_SUBSETS


@pytest.fixture
def root(tmp_path):
    subsets = {"short_m": ("MS", [40, 60]), "long_h": ("H", [3000]), "fred_md": ("MS", [300]),
               "synth_tight_H": ("H", [2000, 2000])}
    manifest = {}
    for name, (freq, lengths) in subsets.items():
        values = np.concatenate([np.arange(n, dtype=np.float32) for n in lengths])
        np.save(tmp_path / f"{name}.values.npy", values)
        np.save(tmp_path / f"{name}.offsets.npy", np.concatenate([[0], np.cumsum(lengths)]))
        manifest[name] = {"subset": name, "n_obs": int(values.size), "freq": freq}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_dev_subsets_stay_out_of_training(root):
    names = train_subsets(root)
    assert "fred_md" not in names and not set(names) & set(DEV_SUBSETS)


def batches(root, n=20, **kw):
    data = WindowBatches(root, 8, 256, max_horizon=64, patch=16, **kw)
    it = iter(data)
    return [next(it) for _ in range(n)]


def test_target_follows_context_without_overlap(root):
    for x, y, _ in batches(root, p_decimate=0.0, p_flip=0.0, p_synth=0.5):
        for c, t in zip(x.numpy(), y.numpy()):
            c, t = c[np.isfinite(c)], t[np.isfinite(t)]
            assert len(c) and len(t)
            assert t[0] == c[-1] + 1
            assert np.all(np.diff(t) == 1)


def test_decimated_windows_keep_order_and_scale(root):
    seen = set()
    for x, y, w in batches(root, p_decimate=1.0, p_flip=0.0):
        for c, t, s in zip(x.numpy(), y.numpy(), w.numpy()):
            c, t = c[np.isfinite(c)], t[np.isfinite(t)]
            assert t[0] > c[-1]
            seen.add(round(float(s), 3))
    assert len(seen) > 2


def test_flip_negates_context_and_target_together(root):
    for x, y, _ in batches(root, p_decimate=0.0, p_flip=1.0):
        assert np.nanmax(x.numpy()) <= 0 and np.nanmax(y.numpy()) <= 0


def test_short_series_are_sampled(root):
    assert any(np.isfinite(x.numpy()).sum(1).min() < 60 for x, _, _ in batches(root))


def test_synthetic_share_is_respected(root):
    data = WindowBatches(root, 8, 256, p_synth=1.0)
    data._setup()
    rng = np.random.default_rng(0)
    picks = [data._subset(rng) for _ in range(200)]
    assert sum(p.name.startswith("synth") for p in picks) > 50
    assert all(p.rate in ("H", "M") for p in picks)
