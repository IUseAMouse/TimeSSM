"""Load a TimeSSM checkpoint into a model built by `build_from_config`.

Same contract as `timejepa.evaluation.loading.load_checkpoint` for the
checkpoints this repository produces (Lightning format, `model.` prefix):
the target-encoder copy and the RevIN runtime buffers are skipped, every
other key must match, and a missing or unexpected key on the model's core
is an error, never a warning.
"""

from typing import Union

import torch

SKIPPED = ("target_encoder",)
RUNTIME_BUFFERS = (".mean", ".std")


def load_checkpoint(model: torch.nn.Module, path: str,
                    device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state.items():
        key = key.replace("model.", "", 1) if key.startswith("model.") else key
        key = key.replace("_orig_mod.", "")
        if any(s in key for s in SKIPPED):
            continue
        if "revin" in key and key.endswith(RUNTIME_BUFFERS):
            continue
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing = [k for k in missing
               if not any(s in k for s in SKIPPED)
               and not ("revin" in k and k.endswith(RUNTIME_BUFFERS))]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing {missing[:6]}, unexpected {unexpected[:6]}")
    return model.to(device).eval()
