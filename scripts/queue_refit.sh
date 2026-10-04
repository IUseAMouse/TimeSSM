#!/usr/bin/env bash
# Arm R1 (2026-10-04): refit the last linear layer of the 2.5M wide champion
# 1.2841 on stored features, then evaluate the refit checkpoint bare and with
# the RateIN-up stack. Waits for a marker line (default: the lookback queue).
#   nohup scripts/queue_refit.sh > logs/queue_refit.out 2>&1 &
#   WAIT_FILE=logs/queue_p0.out WAIT_FOR="phase 0 queue done" nohup scripts/queue_refit.sh > logs/queue_refit.out 2>&1 &
#   REFIT_ARGS="--per-dataset 8192 --steps 32" ...   extra flags for refit_last_layer.py; the rows
#       are spread over the three GPUs (--gb-per-gpu 15 by default, 0.77 GB per million rows)
set -u
cd "$(dirname "$0")/.."
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False; CK=epoch00_valloss1.2841
OUT=checkpoints/timessm_mini_v3_wide_refit/pretrain_False
UP="+tta_flip=true +ratein=mix +ratein_pool=true +ratein_k_up=2x3x4 +ratein_min_bt=4 +ratein_bt_windows=4"
[ -f "$D/$CK.ckpt" ] || { echo "missing checkpoint"; exit 2; }
WAIT_FILE=${WAIT_FILE:-logs/queue_lb.out}
until grep -q "${WAIT_FOR:-lookback queue done}" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
mkdir -p logs
echo "== $(date '+%F %T') refit"
python scripts/refit_last_layer.py --checkpoint "$D/$CK.ckpt" --out "$OUT" ${REFIT_ARGS:-} > logs/refit.out 2>&1
grep -E "features|pinball|median MAE|weight change|written|no checkpoint" logs/refit.out
[ -f "$OUT/$CK-refit.ckpt" ] || { echo "no refit checkpoint: nothing to evaluate"; echo "== refit queue done"; exit 0; }
echo "== $(date '+%F %T') evals of the refit checkpoint"
CUDA_VISIBLE_DEVICES=0 STACK="$UP" bash scripts/eval_checkpoints_ssm.sh "$OUT" +gift_batch_size=32 > logs/refit_up.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK="" bash scripts/eval_checkpoints_ssm.sh "$OUT" +gift_batch_size=32 > logs/refit_nu.out 2>&1 &
wait
grep -h "vs_official" logs/refit_up.out logs/refit_nu.out
echo "== $(date '+%F %T') refit queue done"
