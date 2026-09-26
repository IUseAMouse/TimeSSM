#!/usr/bin/env bash
# Quantile temperature (CQR gamma) for a TimeSSM checkpoint, with TimeJEPA's
# calibrate_quantiles.py (G4.2) on this repo's config (plan B2, 2026-09-26).
#
#   scripts/calibrate_ssm.sh <checkpoint> [--flip] [--horizon 256] [--config-name ssm_mini_v3]
#   -> ../TimeJEPA/evaluation/calibration/gamma_<stem>[_flip].json, then
#   STACK="+tta_flip=true +ratein=mix +ratein_pool=true +quantile_gamma=<json>" ONLY=<stem> scripts/eval_checkpoints_ssm.sh <dir>
#
# Read `coverage_before` per dataset in the JSON first: ~0.80 in distribution
# means the GIFT under-coverage is shift (gamma ~ 1, the arm is dead, as on
# TimeJEPA); ~0.70 means the fan itself is narrow (gamma > 1, evaluate).
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
