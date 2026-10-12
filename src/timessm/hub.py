"""Load TimeSSM weights from a Hugging Face repository or a local directory.

    from timessm import from_pretrained
    model = from_pretrained("IUseAMouse/timessm-2.5m")

A pretrained directory holds `config.json` (the `model` block of the Hydra
config, as `build_from_config` reads it) and `model.safetensors` (the
model's state_dict, no optimizer, no target-encoder copy). Optional files
ride along for the evaluation harness: the calibration factors of the fan
and the frequency table. `scripts/export_hf.py` writes such a directory from
a training checkpoint.
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import OmegaConf

from .model import build_from_config

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def _resolve(repo_or_path: str, revision: Optional[str], token: Optional[str]) -> Path:
    p = Path(repo_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_or_path, revision=revision, token=token,
                                  allow_patterns=[CONFIG_FILE, WEIGHTS_FILE, "*.json", "*.yaml"]))


def from_pretrained(repo_or_path: str, device: Union[str, torch.device] = "cpu",
                    revision: Optional[str] = None, token: Optional[str] = None):
    """Build the model described by `config.json` and load `model.safetensors`.
    `repo_or_path` is a Hugging Face repository id or a local directory."""
    from safetensors.torch import load_file
    root = _resolve(repo_or_path, revision, token)
    config = json.loads((root / CONFIG_FILE).read_text())
    model = build_from_config(OmegaConf.create({"model": config}))
    state = load_file(str(root / WEIGHTS_FILE))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not ("revin" in k and k.endswith((".mean", ".std")))]
    if missing or unexpected:
        raise RuntimeError(f"weights/model mismatch: missing {missing[:6]}, unexpected {unexpected[:6]}")
    return model.to(torch.device(device)).eval()


def export(model, config, out_dir: Union[str, Path]) -> Path:
    """Write `config.json` and `model.safetensors` for `from_pretrained`.
    `config` is the `model` block (a DictConfig or a dict)."""
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    block = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    (out / CONFIG_FILE).write_text(json.dumps(block, indent=2) + "\n")
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
             if not ("revin" in k and k.endswith((".mean", ".std")))}
    save_file(state, str(out / WEIGHTS_FILE))
    return out
