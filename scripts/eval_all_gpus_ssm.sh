#!/usr/bin/env bash
# Evaluate every checkpoint of a run on GIFT-Eval using ALL visible GPUs
# (2026-09-25): the checkpoints are dealt round-robin to the GPUs, each GPU
# runs its share one after the other through eval_checkpoints_ssm.sh (ONLY=),
# then a final single pass over the whole directory re-runs whatever failed
# (the cache keeps successes only, so an OOM'd config is recomputed, not
# skipped) and prints the table with the n_cfg column: a row below 97 is
# flagged with * and is NOT a result.
#
#   scripts/eval_all_gpus_ssm.sh <checkpoint_dir> [extra hydra flags...]
#   GPUS="0 1 2" EVAL_CONFIG=ssm_mid_v3_eval scripts/eval_all_gpus_ssm.sh checkpoints/timessm_mid_v3_zs/pretrain_False +gift_batch_size=48
#   STACK="+ratein=backtest +ratein_pool=true" scripts/eval_all_gpus_ssm.sh <dir>       # other flag sets, same layout
set -u
if [ $# -lt 1 ]; then echo "usage: $0 <checkpoint_dir> [extra hydra flags...]" >&2; exit 2; fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
DIR="$1"; shift
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
read -r -a GPU_ARR <<< "$GPUS"
mapfile -t CKPTS < <(ls -tr "$DIR"/*.ckpt 2>/dev/null | grep -v '/last[^/]*\.ckpt$')
[ ${#CKPTS[@]} -gt 0 ] || { echo "no checkpoint in $DIR"; exit 1; }
RUN=$(basename "$(dirname "$DIR")")
mkdir -p "$HERE/logs"
echo "== $(date '+%F %T') ${#CKPTS[@]} checkpoints on ${#GPU_ARR[@]} GPU (${GPU_ARR[*]}): ${STACK:-official stack} $*"
declare -a PIDS=()
for k in "${!GPU_ARR[@]}"; do
  g=${GPU_ARR[$k]}
  (
    for ((i = k; i < ${#CKPTS[@]}; i += ${#GPU_ARR[@]})); do
      STEM=$(basename "${CKPTS[$i]}" .ckpt)
      echo "[gpu $g] $(date '+%T') $STEM"
      CUDA_VISIBLE_DEVICES=$g ONLY="$STEM" bash "$HERE/scripts/eval_checkpoints_ssm.sh" "$DIR" "$@" \
        > "$HERE/logs/eval_${RUN}_gpu${g}_${STEM}.out" 2>&1
      grep -E "configs:|vs_official" "$HERE/logs/eval_${RUN}_gpu${g}_${STEM}.out" | sed "s/^/[gpu $g] /"
    done
  ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
echo "== $(date '+%F %T') parallel pass done; final pass on GPU ${GPU_ARR[0]}: recompute failures, print the table"
CUDA_VISIBLE_DEVICES=${GPU_ARR[0]} bash "$HERE/scripts/eval_checkpoints_ssm.sh" "$DIR" "$@" 2>&1 | tail -n $(( ${#CKPTS[@]} + 6 ))
