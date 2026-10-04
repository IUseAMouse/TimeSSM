#!/usr/bin/env bash
# Night of 2026-10-04 on the 2.5M wide champion 1.2841, behind the selector
# queue (scripts/queue_selector.sh). Nothing here depends on a decision.
#   nohup scripts/queue_phase0.sh > logs/queue_p0.out 2>&1 &
# 1. now, CPU only: phase 0 of the MASE plan (diagnose_median.py sn and data);
# 2. once logs/queue_s.out says the selector queue is done: `flat` on GPU 0,
#    then the RateIN-up stack on EVERY wide checkpoint, 3 GPUs (the band of
#    the model card: the last checkpoints, not the best point).
set -u
cd "$(dirname "$0")/.."
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False; CK=epoch00_valloss1.2841
R=evaluation/timessm_mini_v3_zs/$CK
UP="+tta_flip=true +ratein=mix +ratein_pool=true +ratein_k_up=2x3x4 +ratein_min_bt=4 +ratein_bt_windows=4"
STACK_DIR=$R/gift_flip_ratein-mix-pool-up234-bt4-w4
[ -d "$STACK_DIR" ] || STACK_DIR=$R/gift_flip_ratein-mix-pool
[ -f "$D/$CK.ckpt" ] && [ -d "$STACK_DIR" ] && [ -d "$R/gift" ] || { echo "missing checkpoint, stack dir or bare dir"; exit 2; }
mkdir -p logs
echo "== $(date '+%F %T') phase 0 on CPU (stack dir: $STACK_DIR)"
python scripts/diagnose_median.py sn "$STACK_DIR" > logs/diag_sn_stack.txt 2>&1
python scripts/diagnose_median.py sn "$R/gift" > logs/diag_sn_nu.txt 2>&1
python scripts/diagnose_median.py data --run "$STACK_DIR" > logs/diag_data.txt 2>&1
echo "== $(date '+%F %T') sn and data done; waiting for the selector queue"
until grep -q "selector queue done" logs/queue_s.out 2>/dev/null; do sleep 60; done
echo "== $(date '+%F %T') flat"
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_median.py flat --checkpoint "$D/$CK.ckpt" > logs/diag_flat.txt 2>&1
echo "== $(date '+%F %T') RateIN-up stack on every wide checkpoint"
STACK="$UP" bash scripts/eval_all_gpus_ssm.sh "$D" +gift_batch_size=32 > logs/band_wide_up.out 2>&1
tail -n 9 logs/band_wide_up.out
echo "== $(date '+%F %T') phase 0 queue done"
