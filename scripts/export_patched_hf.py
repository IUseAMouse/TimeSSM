"""Write a Hugging Face model directory from a train_patched.py run.

    python scripts/export_patched_hf.py --config runs/timessm_10m_d160k/config.yaml \
        --checkpoint runs/timessm_10m_d160k/timessm_10m_d160k.ckpt --out export/timessm-10m \
        --card docs/model_card_timessm_10m.md

The directory holds config.json (the `model` block of the run, with its `builder`) and
model.safetensors (the averaged weights the run evaluates), read by `timessm.from_pretrained`,
the same weights as <name>.ckpt for the evaluation harness, and the model card as README.md.
Nothing is uploaded.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timessm import from_pretrained  # noqa: E402
from timessm.hub import export  # noqa: E402
from timessm.patched import build_from_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="config.yaml of the run")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="model name written in config.json (default: the run's)")
    ap.add_argument("--card", help="model card, copied as README.md")
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    if a.name:
        cfg.model.name = a.name
    model = build_from_config(cfg).eval()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items()})
    out = export(model, cfg.model, a.out)
    # the same weights in the checkpoint format of the evaluation harness (scripts/eval_ssm.sh)
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}},
               out / f"{cfg.model.name}.ckpt")
    if a.card:
        (out / "README.md").write_text(Path(a.card).read_text())

    # round trip: the exported weights must reproduce the checkpoint bit for bit
    back = from_pretrained(str(out))
    torch.manual_seed(0)
    x = torch.randn(2, 512)
    with torch.no_grad():
        q = model.forecast(x, 64, w=0.5)["quantiles_denorm"]
        assert torch.equal(q, back.forecast(x, 64, w=0.5)["quantiles_denorm"]), \
            "exported weights do not reproduce the checkpoint"
    n = sum(p.numel() for p in back.parameters())
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"{out}: {sorted(f.name for f in out.iterdir())}, {n} parameters, {size / 1e6:.1f} MB, "
          "round trip exact")


if __name__ == "__main__":
    main()
