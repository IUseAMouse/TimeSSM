"""
Multivariate items: scripts/build_groups.py finds the variates of raw items back among the
stored series (dropped, ambiguous and constant ones left out), WindowBatches draws aligned
items and synthetic covariate items, the training loss reads them, and the multivariate
dev scores them.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from timessm.corpus import (DEV_MULTI, WindowBatches, covariate_target,  # noqa: E402
                            load_subsets, train_subsets)
from timessm.groups import aligned, load_groups, span  # noqa: E402
from timessm.patched import FUTURE_COVARIATE, PAST_COVARIATE, TARGET, PatchedSSM  # noqa: E402

spec = importlib.util.spec_from_file_location("build_groups", ROOT / "scripts" / "build_groups.py")
BG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BG)

T = 1200


def ramp(c: int, start: int = 0) -> np.ndarray:
    """Variate c of an item: the time index plus 1000 c, so that aligned variates differ by
    a constant and a shift shows."""
    x = (np.arange(T) + 1000.0 * c + 1.0).astype(np.float32)
    x[:start] = np.nan
    return x


def store(corpus: Path, name: str, rows, freq: str, manifest: dict, dropped=0):
    rows = [np.asarray(r, np.float32) for r in rows]
    np.save(corpus / f"{name}.values.npy", np.concatenate(rows))
    np.save(corpus / f"{name}.offsets.npy", np.concatenate([[0], np.cumsum([len(r) for r in rows])]))
    manifest[name] = {"subset": name, "n_obs": int(sum(len(r) for r in rows)), "freq": freq,
                      **({"dropped": dropped} if dropped else {})}


def write_arrow(path: Path, targets):
    path.mkdir(parents=True)
    table = pa.table({"target": [t.tolist() for t in targets]})
    with pa.OSFile(str(path / "data-00000-of-00001.arrow"), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def lotsa_cut(ch):
    obs = np.flatnonzero(np.isfinite(ch))
    return ch[obs[0]:obs[-1] + 1] if len(obs) >= 16 else None


@pytest.fixture
def sources(tmp_path):
    corpus, raw, caches = tmp_path / "v10", tmp_path / "raw", tmp_path / "raw_d"
    for d in (corpus, raw, caches / "isd"):
        d.mkdir(parents=True)
    manifest = {}
    # LOTSA multivariate subset: 4 items of 3 variates, variate 1 starts late; the stored
    # copy lost one series (an overlap removal) and dropped nothing else.
    items = [np.stack([ramp(c, start=40 * c) + 10000 * i for c in range(3)]) for i in range(4)]
    # A fourth variate with 10 points: the corpus builder drops it, not counted as missing.
    items = [np.concatenate([it, np.r_[np.full(T - 10, np.nan), np.arange(10.0)][None]])
             .astype(np.float32) for it in items]
    write_arrow(raw / "multi_h", items)
    rows = [lotsa_cut(ch) for it in items for ch in it[:3]]
    store(corpus, "multi_h", rows[:4] + rows[5:], "H", manifest, dropped=1)
    # A univariate LOTSA subset: skipped.
    write_arrow(raw / "uni_d", [np.arange(100, dtype=np.float32)])
    store(corpus, "uni_d", [np.arange(100, dtype=np.float32)], "D", manifest)
    # ISD caches: 3 stations x 4 channels; station 2's pressure is constant (dropped).
    grids = []
    for st in range(3):
        g = np.stack([ramp(c, start=5 * st) + 50000 * st for c in range(4)])
        if st == 2:
            g[2] = 7.0
        np.save(caches / "isd" / f"st{st}.npy", g)
        grids.append(g)
    for c, name in enumerate(BG.ISD_CHANNELS):
        kept = [lotsa_cut(g[c]) for g in grids if np.nanstd(g[c]) > 0]
        store(corpus, f"isd_{name}_H", kept, "H", manifest)
    # Wind farm: 2 turbines x 4 channels, by index; and the dev farm (kelmarsh).
    for farm in ("penmanshiel", "kelmarsh"):
        for c, ch in enumerate(BG.WIND_CHANNELS):
            store(corpus, f"{farm}_{ch}", [ramp(c) + 70000 * t for t in range(2)], "10T", manifest)
    # A duplicated series in a univariate store: ambiguous, never matched.
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    return corpus, raw, caches, items, grids


def test_build_groups_finds_the_variates(sources, tmp_path):
    corpus, raw, caches, items, grids = sources
    out = tmp_path / "v10m"
    summary = BG.build(corpus, out, raw, caches)
    assert summary["sources"]["multi_h"] == {"variates": 12, "found": 11, "ambiguous": 0,
                                             "items": 4}
    assert summary["sources"]["isd"]["found"] == 11 and summary["sources"]["isd"]["items"] == 3
    assert summary["sources"]["penmanshiel"]["items"] == 2
    assert "uni_d" not in summary["sources"]
    assert (out / "multi_h.values.npy").is_symlink() and (out / "groups.npz").exists()
    subsets = load_subsets(out, json.loads((out / "manifest.json").read_text()))
    groups = load_groups(out, subsets)
    multi = [g for g in groups if g[0][0].name == "multi_h"]
    # Item 1 lost its variate 1 (the removed series): two members, shifts 0 and 80.
    assert [(s.name, shift) for s, _, shift in multi[1]] == [("multi_h", 0), ("multi_h", 80)]
    for it, raw_item in zip(multi, items):
        start, end = span(it)
        got = aligned(it, start, end)
        cs = [shift // 40 for _, _, shift in it]       # variate c starts at 40 c
        np.testing.assert_array_equal(got, raw_item[cs][:, start:end])
    # ISD: members from four subsets on one station axis.
    isd = [g for g in groups if g[0][0].name.startswith("isd")]
    assert sorted(len(g) for g in isd) == [3, 4, 4]


def test_held_out_subsets_drop_their_members(sources, tmp_path):
    corpus, raw, caches, _, _ = sources
    out = tmp_path / "v10m"
    BG.build(corpus, out, raw, caches)
    names = train_subsets(out, multivariate=True)
    assert not set(names) & set(DEV_MULTI) and "kelmarsh_power" in train_subsets(out)
    groups = load_groups(out, load_subsets(out, names))
    assert not any(m[0].name.startswith("kelmarsh") for g in groups for m in g)


def test_wrong_cut_rule_is_refused(sources, tmp_path):
    corpus, raw, caches, items, _ = sources
    manifest = json.loads((corpus / "manifest.json").read_text())
    store(corpus, "multi_h", [ch[::-1] for it in items for ch in it], "H", manifest)
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="cut rule"):
        BG.build(corpus, tmp_path / "bad", raw, None)


def test_ambiguous_series_match_nothing(tmp_path):
    manifest = {}
    store(tmp_path, "dup", [np.arange(50.0), np.arange(50.0), np.arange(60.0)], "H", manifest)
    index = BG.Index(tmp_path, ["dup"])
    assert index.find(np.arange(50.0, dtype=np.float32)) is None
    assert index.is_ambiguous(np.arange(50.0, dtype=np.float32))
    tail = np.full(10, np.nan)
    grid = np.stack([np.r_[np.arange(50.0), tail], np.r_[np.arange(50.0) + 7, tail],
                     np.arange(60.0)]).astype(np.float32)
    found, stats = BG.match_items([grid], index)
    assert stats == {"variates": 3, "found": 1, "ambiguous": 1, "items": 0}
    assert index.find(np.r_[np.nan, np.arange(60.0)].astype(np.float32)) == ("dup", 2, 1)


@pytest.fixture
def v10m(sources, tmp_path):
    corpus, raw, caches, _, _ = sources
    out = tmp_path / "v10m"
    BG.build(corpus, out, raw, caches)
    return out


def multi_batches(root, n=12, **kw):
    kw = {"p_decimate": 0.0, "p_flip": 0.0, "p_multi": 1.0, "p_cov_synth": 0.0, **kw}
    it = iter(WindowBatches(root, 16, 256, max_horizon=64, patch=16, groups=2, **kw))
    return [next(it) for _ in range(n)]


@pytest.mark.parametrize("kw", [{}, {"p_decimate": 1.0}, {"p_flip": 1.0}])
def test_items_are_aligned(v10m, kw):
    for x, y, w, roles, valid in multi_batches(v10m, **kw):
        B, V, L = x.shape
        assert B * V <= 16 and y.shape[:2] == (B, V) and roles.shape == valid.shape == (B, V)
        for b in range(B):
            assert (roles[b][valid[b]] == TARGET).any()
            full = torch.cat([x[b], y[b]], 1)
            rows = [r for r in range(V) if valid[b, r]]
            ref = full[rows[0]]
            for r in rows[1:]:
                d = (full[r] - ref)[torch.isfinite(full[r] - ref)]
                assert len(d) and torch.allclose(d, d[0].expand_as(d)), "variates shifted"
            if kw.get("p_flip"):
                assert torch.nan_to_num(full[rows]).max() <= 0
            if not kw:
                fx, fy = x[b, rows[0]], y[b, rows[0]]
                fx, fy = fx[torch.isfinite(fx)], fy[torch.isfinite(fy)]
                assert fy[0] == fx[-1] + 1


def test_synthetic_covariate_items(v10m):
    for x, y, w, roles, valid in multi_batches(v10m, p_cov_synth=1.0):
        assert 2 <= x.shape[1] <= 5 and valid.all()
        assert (roles[:, 0] == TARGET).all()
        assert set(roles[:, 1:].unique().tolist()) <= {PAST_COVARIATE, FUTURE_COVARIATE}


def test_covariate_target_is_the_stated_sum():
    rng = np.random.default_rng(0)
    base = (np.sin(np.arange(200) / 5) * 3 + 10).astype(np.float32)
    covs = np.stack([np.arange(200.0), (np.arange(200) % 7 == 0).astype(float)]).astype(np.float32)
    y, spec = covariate_target(np.random.default_rng(1), base, covs, 150, 24)
    sd_b = base[:150].std()
    b_hat = (base - base[:150].mean()) / sd_b
    effect = np.zeros(200)
    for c, (kind, lag, a) in zip(covs, spec):
        z = (c - c[:150].mean()) / c[:150].std()
        z = np.concatenate([np.full(lag, z[0]), z[:200 - lag]]) if lag else z
        effect += a * (z, np.tanh(2 * z), (z > 0).astype(float), z * b_hat)[kind]
    np.testing.assert_allclose(y, base + sd_b * effect, rtol=1e-4, atol=1e-3)


def test_multivariate_loss(v10m):
    import train_patched
    torch.manual_seed(0)
    model = PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4, n_heads=4,
                       causal=True, loops=2, variate_layers=(1,), variate_qk=16).train()
    levels = torch.tensor(model.levels)
    t = OmegaConf.create({"loss": "asinh", "context_loss": 1.0, "context_skip": 2})
    x, y, w, roles, valid = multi_batches(v10m, n=1)[0]
    roles[:, :] = TARGET
    roles[:, 1] = PAST_COVARIATE
    loss = train_patched.step_loss(model, x, y, w, levels, t, 16, None, roles, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.var_att["1"].q.weight.grad.abs().sum() > 0
    with torch.no_grad():
        y2 = y.clone()
        y2[:, 1] = y2[:, 1] * 3 + 5              # a past covariate's horizon is never read
        same = train_patched.step_loss(model, x, y2, w, levels, t, 16, None, roles, valid)
        roles[:, 1] = FUTURE_COVARIATE           # a future covariate's horizon is read
        a = train_patched.step_loss(model, x, y, w, levels, t, 16, None, roles, valid)
        b = train_patched.step_loss(model, x, y2, w, levels, t, 16, None, roles, valid)
    assert torch.equal(loss.detach(), same) and not torch.equal(a, b)


def test_dev_multi_score(sources, tmp_path):
    corpus, raw, caches, _, _ = sources
    out = tmp_path / "v10m"
    BG.build(corpus, out, raw, caches)
    import dev_eval
    torch.manual_seed(0)
    model = PatchedSSM(input_length=256, patch=16, d_model=32, n_layers=2, d_state=4, n_heads=4,
                       causal=True, loops=2, variate_layers=(1,), variate_qk=16).eval()
    fams = {fam: roles for fam, _, roles, _, _ in dev_eval.dev_items(str(out))}
    assert set(fams) == {"kelmarsh"}
    assert fams["kelmarsh"][0].tolist() == [TARGET, PAST_COVARIATE, PAST_COVARIATE,
                                            FUTURE_COVARIATE]
    score = dev_eval.dev_multi_score(model, str(out))
    assert set(score) == {"joint", "single"}
    assert np.isfinite(score["joint"]["all"]) and np.isfinite(score["single"]["all"])
