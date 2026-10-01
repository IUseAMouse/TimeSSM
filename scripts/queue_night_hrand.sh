#!/usr/bin/env bash
# Night queue (2026-10-01): everything that can run on the three GPUs while
# the user sleeps, in waves; each wave waits on the PIDs of its own children.
#   1. the 5 checkpoints of the B1 arm (ssm_mid_v3_hrand) on all GPUs, official stack
#      + the CQR calibration of its LAST checkpoint on CPU, in parallel
#   2. on the 2.5M wide 1.2841, one per GPU: nu, flip, RateIN-up ablation (a) bt_windows=4 alone
#   3. on the 2.5M wide 1.2841: ablation (b) k_up alone, (c) min_bt alone, and the combined
#      RateIN-up + gamma
#   4. the B1 last checkpoint with RateIN-up + its own gamma (second round)
#
#   nohup scripts/queue_night_hrand.sh > logs/queue_night.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
TIMEJEPA=${TIMEJEPA:-$(cd ../TimeJEPA && pwd)}
D25=checkpoints/timessm_mini_v3_wide_zs/pretrain_False
CK25=epoch00_valloss1.2841
G25=$TIMEJEPA/evaluation/calibration/gamma_${CK25}_flip.json
DH=checkpoints/timessm_mid_v3_hrand_zs/pretrain_False
S="+tta_flip=true +ratein=mix +ratein_pool=true"
UP="+ratein_k_up=2x3x4 +ratein_min_bt=4 +ratein_bt_windows=4"
[ -f "$D25/$CK25.ckpt" ] || { echo "missing $D25/$CK25.ckpt"; exit 2; }
[ -f "$G25" ] || { echo "missing $G25"; exit 2; }
ls "$DH"/epoch00_*.ckpt > /dev/null 2>&1 || { echo "no checkpoint in $DH"; exit 2; }
LAST=$(ls -t "$DH"/epoch00_*.ckpt | head -1); LAST_STEM=$(basename "$LAST" .ckpt)
mkdir -p logs
echo "== $(date '+%F %T') wave 1: B1 checkpoints on all GPUs + calibration of $LAST_STEM on CPU"
EVAL_CONFIG=ssm_mid_v3_hrand_eval bash scripts/eval_all_gpus_ssm.sh "$DH" +gift_batch_size=48 > logs/eval_hrand.out 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES="" bash scripts/calibrate_ssm.sh "$LAST" --flip --config-name ssm_mid_v3_hrand > logs/calib_hrand.out 2>&1 &
P2=$!
wait $P1 $P2
echo "== $(date '+%F %T') wave 1 done"; tail -n 8 logs/eval_hrand.out
GH=$TIMEJEPA/evaluation/calibration/gamma_${LAST_STEM}_flip.json
echo "== $(date '+%F %T') wave 2: 2.5M wide nu / flip / ablation (a) bt_windows"
CUDA_VISIBLE_DEVICES=0 STACK="" ONLY=$CK25 bash scripts/eval_checkpoints_ssm.sh $D25 +gift_batch_size=32 > logs/nu25.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK="+tta_flip=true" ONLY=$CK25 bash scripts/eval_checkpoints_ssm.sh $D25 +gift_batch_size=32 > logs/flip25.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 STACK="$S +ratein_bt_windows=4" ONLY=$CK25 bash scripts/eval_checkpoints_ssm.sh $D25 +gift_batch_size=32 > logs/abl_w4.out 2>&1 &
wait
echo "== $(date '+%F %T') wave 2 done"; grep -h "vs_official" logs/nu25.out logs/flip25.out logs/abl_w4.out
echo "== $(date '+%F %T') wave 3: ablation (b) k_up, (c) min_bt, combined up + gamma"
CUDA_VISIBLE_DEVICES=0 STACK="$S +ratein_k_up=2x3x4" ONLY=$CK25 bash scripts/eval_checkpoints_ssm.sh $D25 +gift_batch_size=32 > logs/abl_kup.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK="$S +ratein_min_bt=4" ONLY=$CK25 bash scripts/eval_checkpoints_ssm.sh $D25 +gift_batch_size=32 > logs/abl_minbt.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 STACK="$S $UP +quantile_gamma=$G25" ONLY=$CK25 bash scripts/eval_checkpoints_ssm.sh $D25 +gift_batch_size=32 > logs/up_gamma25.out 2>&1 &
wait
echo "== $(date '+%F %T') wave 3 done"; grep -h "vs_official" logs/abl_kup.out logs/abl_minbt.out logs/up_gamma25.out
echo "== $(date '+%F %T') wave 4: B1 last checkpoint $LAST_STEM with RateIN-up + its gamma"
if [ -f "$GH" ]; then
  CUDA_VISIBLE_DEVICES=0 EVAL_CONFIG=ssm_mid_v3_hrand_eval STACK="$S $UP +quantile_gamma=$GH" ONLY=$LAST_STEM \
    bash scripts/eval_checkpoints_ssm.sh "$DH" +gift_batch_size=48 > logs/hrand_up_gamma.out 2>&1 &
else
  echo "no gamma for $LAST_STEM (see logs/calib_hrand.out): up without gamma"
fi
CUDA_VISIBLE_DEVICES=1 EVAL_CONFIG=ssm_mid_v3_hrand_eval STACK="$S $UP" ONLY=$LAST_STEM \
  bash scripts/eval_checkpoints_ssm.sh "$DH" +gift_batch_size=48 > logs/hrand_up.out 2>&1 &
wait
echo "== $(date '+%F %T') night queue done"; grep -h "vs_official" logs/hrand_up.out logs/hrand_up_gamma.out 2>/dev/null
