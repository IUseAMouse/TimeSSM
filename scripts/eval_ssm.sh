#!/usr/bin/env bash
# Evaluate a TimeSSM checkpoint on GIFT-Eval with TimeJEPA's harness, from
# this repository (results in ./evaluation/<model.name>/<ckpt>/gift<tag>/).
#
#   scripts/eval_ssm.sh <checkpoint> [hydra flags...]
#   EVAL_CONFIG=timessm_2.5m_gift scripts/eval_ssm.sh epoch00_valloss1.2814.ckpt   # the released number
#   scripts/eval_ssm.sh epoch00_valloss1.2814.ckpt +freq_delta=true                # bare, frequency given
#   scripts/eval_ssm.sh <ckpt> +freq_delta=true +tta_flip=true +ratein=oracle       # best rate per config, diagnostic
#
# EVAL_CONFIG=<name> picks another eval config (default ssm_mini_v3_freq_eval,
# the published checkpoint's architecture; it expects +freq_delta=true);
# TIMEJEPA=<path> overrides the sibling checkout. The GIFT data are read from
# TimeJEPA (+gift_data_dir), the config from this repo (absolute config path).
set -eu
if [ $# -lt 1 ]; then
  echo "usage: $0 <checkpoint> [hydra flags...]" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
TIMEJEPA=${TIMEJEPA:-"$HERE/../TimeJEPA"}
# timessm is imported by TimeJEPA's harness through model.builder: make it
# importable from any environment (the timejepa venv has no timessm package).
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
CK="$1"; shift
PYTHONUNBUFFERED=1 python "$TIMEJEPA/scripts/evaluate_gift.py" \
  --config-path "$HERE/configs" --config-name "${EVAL_CONFIG:-ssm_mini_v3_freq_eval}" \
  "+checkpoint_path=$CK" "+gift_data_dir=$TIMEJEPA/data/gift_eval" "$@"
