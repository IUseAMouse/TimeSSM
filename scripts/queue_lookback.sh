#!/usr/bin/env bash
# P-SSM.10 (2026-10-04): is the context length a second selection axis? Bare
# 2.5M wide champion, 97 configs, behind the phase 0 queue.
#   nohup scripts/queue_lookback.sh > logs/queue_lb.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False; CK=epoch00_valloss1.2841
[ -f "$D/$CK.ckpt" ] || { echo "missing checkpoint"; exit 2; }
until grep -q "phase 0 queue done" logs/queue_p0.out 2>/dev/null; do sleep 60; done
echo "== $(date '+%F %T') lookback evals"
CUDA_VISIBLE_DEVICES=0 STACK="+max_context=256" ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/lb_ctx256.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 STACK='+tta_lookbacks="256,1024"' ONLY=$CK bash scripts/eval_checkpoints_ssm.sh $D +gift_batch_size=32 > logs/lb_avg.out 2>&1 &
wait
grep -h "vs_official" logs/lb_ctx256.out logs/lb_avg.out
echo "== $(date '+%F %T') lookback queue done"
