#!/usr/bin/env bash
# Selector arm S (2026-10-04), inference only, on the 2.5M wide champion 1.2841
# (where the oracle run already exists) - never on the B5 checkpoints, which
# degraded the stack. Waits for a marker line in a log, then two waves.
#   WAIT_FILE=logs/queue_dec.out WAIT_FOR="^done" nohup scripts/queue_selector.sh > logs/queue_s.out 2>&1 &
# Wave 1: backtest (hard choice, 4 windows) for the gap report vs the oracle; mix with 8
#         windows; the bare eval of the 10M B1 last checkpoint (the missing reference for B5).
# Wave 2: mix with 4 windows at tau 0.03 and 0.08; the selection-gap report on CPU.
set -u
cd "$(dirname "$0")/.."
TIMEJEPA=${TIMEJEPA:-$(cd ../TimeJEPA && pwd)}
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False; CK=epoch00_valloss1.2841
R=evaluation/timessm_mini_v3_zs/$CK
DH=checkpoints/timessm_mid_v3_hrand_zs/pretrain_False; CKH=epoch00_valloss3.0559-v1
[ -f "$D/$CK.ckpt" ] && [ -f "$DH/$CKH.ckpt" ] && [ -d "$R/gift_flip_ratein-oracle" ] || { echo "missing checkpoint or oracle dir"; exit 2; }
if [ -n "${WAIT_FILE:-}" ]; then
  until grep -q "${WAIT_FOR:-^done}" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
fi
mkdir -p logs
M="+tta_flip=true +ratein=mix +ratein_pool=true"
echo "== $(date '+%F %T') wave 1"
CUDA_VISIBLE_DEVICES=0 STACK="+tta_flip=true +ratein=backtest +ratein_pool=true +ratein_bt_windows=4" ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/s_bt4.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK="$M +ratein_bt_windows=8" ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/s_mix_w8.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 EVAL_CONFIG=ssm_mid_v3_hrand_eval STACK="" ONLY=$CKH bash scripts/eval_checkpoints_ssm.sh $DH +gift_batch_size=48 > logs/nu_hrand.out 2>&1 &
wait
grep -h "vs_official" logs/s_bt4.out logs/s_mix_w8.out logs/nu_hrand.out
echo "== $(date '+%F %T') wave 2"
CUDA_VISIBLE_DEVICES=0 STACK="$M +ratein_bt_windows=4 +ratein_mix_tau=0.03" ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/s_tau003.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK="$M +ratein_bt_windows=4 +ratein_mix_tau=0.08" ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/s_tau008.out 2>&1 &
BT=$(ls -d "$R"/gift_flip_ratein-bt-pool*w4 2>/dev/null | head -1)
[ -n "$BT" ] && python "$TIMEJEPA/scripts/ratein_selection_gap.py" --bt "$BT" --oracle "$R/gift_flip_ratein-oracle" > logs/s_gap.txt 2>&1 || echo "no backtest dir found for the gap report"
wait
grep -h "vs_official" logs/s_tau003.out logs/s_tau008.out
echo "== $(date '+%F %T') selector queue done"
