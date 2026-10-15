#!/usr/bin/env bash
# Train the RateIN layer around a frozen TimeSSM checkpoint with TimeJEPA's
# trainer, from the TimeMamba repo (config configs/ratein_layer_e2e.yaml,
# checkpoint written under ./checkpoints/ratein_layer/).
#
#   scripts/train_ratein_layer.sh <checkpoint> [hydra overrides...]
#   scripts/train_ratein_layer.sh checkpoints/timessm_mini_v3_freq_zs/pretrain_False/epoch00_valloss1.2814.ckpt
#   scripts/train_ratein_layer.sh <ckpt> ratein_layer.steps=200 ratein_layer.wandb=false      # smoke
#
# CONFIG=<name> picks another config (default ratein_layer_e2e);
# TIMEJEPA=<path> overrides the sibling checkout.
set -eu
if [ $# -lt 1 ]; then
  echo "usage: $0 <checkpoint> [hydra overrides...]" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
TIMEJEPA=${TIMEJEPA:-"$HERE/../TimeJEPA"}
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
CK="$1"; shift
PYTHONUNBUFFERED=1 python "$TIMEJEPA/scripts/train_ratein_layer.py" \
  --config-path "$HERE/configs" --config-name "${CONFIG:-ratein_layer_e2e}" \
  "+checkpoint_path=$CK" "$@"
