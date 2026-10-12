#!/usr/bin/env bash
# Quantile temperature (CQR gamma) for a TimeSSM checkpoint, with TimeJEPA's
# calibrate_quantiles.py on this repo's config.
#
#   scripts/calibrate_ssm.sh <checkpoint> [--flip] [--horizon 256] [--config-name ssm_mini_v3]
#   -> ../TimeJEPA/evaluation/calibration/gamma_<stem>[_flip].json, then
#   STACK="+tta_flip=true +ratein=mix +ratein_pool=true +quantile_gamma=<json>" ONLY=<stem> scripts/eval_checkpoints_ssm.sh <dir>
#
# `coverage_before` per dataset in the JSON tells whether the fan itself is
# narrow (about 0.70, gamma > 1) or the under-coverage comes from shift
# (about 0.80, gamma near 1).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
TIMEJEPA=${TIMEJEPA:-"$HERE/../TimeJEPA"}
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
CK="$1"; shift
CONFIG=ssm_mini_v3
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config-name) CONFIG="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
cd "$TIMEJEPA"
PYTHONUNBUFFERED=1 python scripts/calibrate_quantiles.py --checkpoint "$HERE/$CK" \
  --config-dir "$HERE/configs" --config-name "$CONFIG" \
  --set "data.data_dir=$TIMEJEPA/data/processed/lotsa_v3" "${ARGS[@]}"
