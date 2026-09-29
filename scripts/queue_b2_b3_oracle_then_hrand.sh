#!/usr/bin/env bash
# One queue for the night (2026-09-29): the three inference evals on the 2.5M
# wide champion, one per GPU, then - when ALL THREE have exited - the B1
# random-horizon continuation of the 10M on the three GPUs. The wait is on
# the PIDs of this shell's own children (`wait`), never on a pgrep pattern
# (the previous queue matched its own command line and never fired).
#
#   nohup scripts/queue_b2_b3_oracle_then_hrand.sh > logs/queue.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False
CK=epoch00_valloss1.2841
G=${GAMMA:-/workspace/TimeJEPA/evaluation/calibration/gamma_${CK}_flip.json}
S="+tta_flip=true +ratein=mix +ratein_pool=true"
START=${START:-checkpoints/timessm_mid_v3_frac_zs/pretrain_False/epoch00_valloss3.0571.ckpt}
[ -f "$D/$CK.ckpt" ] || { echo "missing $D/$CK.ckpt"; exit 2; }
[ -f "$G" ] || { echo "missing $G"; exit 2; }
[ -f "$START" ] || { echo "missing $START"; exit 2; }
mkdir -p logs
echo "== $(date '+%F %T') three evals"
CUDA_VISIBLE_DEVICES=0 STACK="$S +quantile_gamma=$G" ONLY=$CK \
  bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/b2.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK="$S +ratein_k_up=2x3x4 +ratein_min_bt=4 +ratein_bt_windows=4" ONLY=$CK \
  bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/b3.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 STACK="+tta_flip=true +ratein=oracle" ONLY=$CK \
  bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/oracle.out 2>&1 &
wait
echo "== $(date '+%F %T') evals done:"
grep -h -E "vs_official|configs:" logs/b2.out logs/b3.out logs/oracle.out
if [ -n "${EVALS_ONLY:-}" ]; then echo "EVALS_ONLY set: no training"; exit 0; fi
echo "== $(date '+%F %T') B1 on the 10M (P-SSM.6b)"
exec bash scripts/train_ssm_loop.sh ssm_mid_v3_hrand ssm-mid-v3-hrand \
  "+training.pretrained_encoder_path=$START" \
  training.schedule_fraction=${FRACTION:-0.04316} \
  training.lr_scheduler.warmup_epochs=${WARMUP:-0.004316}
