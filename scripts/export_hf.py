"""Write a Hugging Face model directory from a TimeSSM training checkpoint.

    python scripts/export_hf.py --checkpoint checkpoints/.../epoch00_valloss1.2814.ckpt \
        --config timessm_2.5m_gift --out export/timessm-2.5m

The directory holds config.json and model.safetensors (read by
`timessm.from_pretrained`), the calibration factors and the frequency table the
evaluation harness uses, the original Lightning checkpoint for exact
reproduction, and the model card as README.md with its Hugging Face header.
Nothing is uploaded: that is `hf auth login` then `hf upload <repo> <dir> . --repo-type model`.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timessm import build_from_config, from_pretrained, load_checkpoint  # noqa: E402
from timessm.hub import export  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

HF_HEADER = """---
license: apache-2.0
library_name: timessm
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - zero-shot
  - state-space-model
  - gift-eval
---

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="timessm_2.5m_gift")
    ap.add_argument("--out", default="export/timessm-2.5m")
    ap.add_argument("--card", default="docs/model_card_timessm_2.5m.md")
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(HERE / "configs")):
        cfg = compose(config_name=args.config)
    model = build_from_config(cfg)
    load_checkpoint(model, args.checkpoint)
    out = export(model, cfg.model, args.out)

    gamma = cfg.get("quantile_gamma")
    if gamma:
        shutil.copy(HERE / gamma, out / "calibration_gamma.json")
    table = HERE / "../TimeJEPA/configs/corpus_v3_frequencies.yaml"
    if table.exists():
        shutil.copy(table, out / "corpus_v3_frequencies.yaml")
    shutil.copy(args.checkpoint, out / Path(args.checkpoint).name)
    card = (HERE / args.card).read_text()
    (out / "README.md").write_text(HF_HEADER + card)
    assets = HERE / "docs" / "assets"
    for name in sorted(p.name for p in assets.glob("*.gif") if f"assets/{p.name}" in card):
        (out / "assets").mkdir(exist_ok=True)
        shutil.copy(assets / name, out / "assets" / name)

    # round trip: the exported weights must reproduce the checkpoint bit for bit
    back = from_pretrained(str(out))
    torch.manual_seed(0)
    x = torch.randn(2, 512)
    with torch.no_grad():
        a = model.forecast(x, 64, w=0.5)["quantiles_denorm"]
        b = back.forecast(x, 64, w=0.5)["quantiles_denorm"]
    assert torch.equal(a, b), "exported weights do not reproduce the checkpoint"
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: {sorted(f.name for f in out.iterdir())}, {size / 1e6:.1f} MB, round trip exact")


if __name__ == "__main__":
    main()
