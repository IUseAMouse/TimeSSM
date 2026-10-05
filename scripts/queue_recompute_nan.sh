#!/usr/bin/env bash
# MASE fix of 2026-10-05 in TimeJEPA's harness (pooled over valid observations,
# as gluonts does): recompute the 15 partly-NaN configs of the published 2.5M
# checkpoint 1.2841 for every stack of its model card; the other 82 configs
# come from the cache. CRPS does not move, MASE may.
#   nohup scripts/queue_recompute_nan.sh > logs/queue_nan.out 2>&1 &
#   GAMMA=<path to the gamma json> ...     if the automatic lookup below finds nothing
set -u
cd "$(dirname "$0")/.."
TIMEJEPA=${TIMEJEPA:-$(cd ../TimeJEPA && pwd)}
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False; CK=epoch00_valloss1.2841
R=evaluation/timessm_mini_v3_zs/$CK
M="+tta_flip=true +ratein=mix +ratein_pool=true"
UP="$M +ratein_k_up=2x3x4 +ratein_min_bt=4 +ratein_bt_windows=4"
[ -f "$D/$CK.ckpt" ] && [ -d "$R" ] || { echo "missing checkpoint or evaluation dir"; exit 2; }
mkdir -p logs
echo "== $(date '+%F %T') before the fix"
for d in "$R"/gift*; do
  [ -f "$d/all_results.csv" ] || continue
  bash "$TIMEJEPA/scripts/requeue_nan_configs.sh" "$d"
done
GAMMA=${GAMMA:-$(ls -t "$R"/gift*up234*_gamma-*/quantile_gamma.json evaluation/calibration/*${CK#epoch00_}*.json \
  "$TIMEJEPA"/evaluation/calibration/*${CK#epoch00_}*.json 2>/dev/null | head -n 1)}
run() {  # run <gpu> <label> <stack flags>
  local gpu=$1 label=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu STACK="$*" ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > "logs/nan_$label.out" 2>&1
  echo "[gpu $gpu] $(date '+%T') $label: $(grep -h 'vs_official' "logs/nan_$label.out" | tail -n 1 | sed 's/^.*MASE/MASE/') $(grep -h 'configs:' "logs/nan_$label.out" | tail -n 1 | sed 's/^ *//')"
}
( run 0 nu ""; run 0 flip "+tta_flip=true" ) &
run 1 stack "$M" &
(
  run 2 up "$UP"
  if [ -n "$GAMMA" ] && [ -f "$GAMMA" ]; then run 2 up_gamma "$UP +quantile_gamma=$GAMMA"
  else echo "[gpu 2] no gamma file found (set GAMMA=<path>): up + gamma not recomputed"; fi
) &
wait
echo "== $(date '+%F %T') recompute queue done"
