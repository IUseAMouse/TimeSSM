#!/usr/bin/env bash
# Keep a training run alive across crashes (2026-09-17): launch, and when the
# process dies with a non-zero exit, resume from the hourly autosave
# (checkpoints/<model>/pretrain_False/last-autosave.ckpt, written by
# train_ssm.py) until the run completes or MAX_RETRIES is reached. Each
# resume takes a fresh data seed so the sampler does not replay the batches
# seen since the autosave (the loop state, optimizer and scheduler come from
# the checkpoint). Every attempt is logged in logs/<run>.attempts.
#
#   scripts/train_ssm_loop.sh ssm_mid_v3 ssm-mid-v3 [extra hydra overrides...]
#   MAX_RETRIES=20 scripts/train_ssm_loop.sh ssm_mid_v3 ssm-mid-v3
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?config name}; RUN=${2:?wandb run name}; shift 2
MAX_RETRIES=${MAX_RETRIES:-30}
FIRST_ONLY=()
MODEL=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/$CONFIG.yaml').model.name)")
SEED=$(python -c "
from hydra import initialize_config_dir, compose; import os
with initialize_config_dir(version_base=None, config_dir=os.path.abspath('configs')):
    print(compose(config_name='$CONFIG').data.seed)")
CK=checkpoints/$MODEL/pretrain_False/last-autosave.ckpt
mkdir -p logs
# User overrides, minus the keys the loop owns (a second +training.resume_ckpt
# would make Hydra refuse the command line on the first relaunch).
EXTRA=()
for a in "$@"; do
  case "$a" in
    +training.resume_ckpt=*|training.resume_ckpt=*|data.seed=*|wandb.run_name=*) FIRST_ONLY+=("$a") ;;
    *) EXTRA+=("$a") ;;
  esac
done
attempt=${START_ATTEMPT:-0}
# An autosave already on disk means a previous process died: resume from it
# straight away (START_ATTEMPT=0 FRESH=1 forces a run from scratch).
if [ "$attempt" -eq 0 ] && [ -f "$CK" ] && [ -z "${FRESH:-}" ] && [ "${#FIRST_ONLY[@]}" -eq 0 ]; then
  attempt=$(( $(grep -c "attempt .* exit" "logs/$RUN.attempts" 2>/dev/null || echo 0) + 1 ))
  echo "== found $CK: resuming as attempt $attempt"
fi
while :; do
  args=(--config-name "$CONFIG" "${EXTRA[@]}")
  if [ "$attempt" -gt 0 ]; then
    [ -f "$CK" ] || { echo "attempt $attempt: no $CK to resume from, STOP"; exit 2; }
    args+=(+training.resume_ckpt="$CK" data.seed=$((SEED + attempt)) wandb.run_name="$RUN-r$attempt")
  else
    args+=(wandb.run_name="$RUN")
    for a in "${FIRST_ONLY[@]}"; do case "$a" in wandb.run_name=*) ;; *) args+=("$a") ;; esac; done
  fi
  echo "== $(date '+%F %T') attempt $attempt: python scripts/train_ssm.py ${args[*]}" | tee -a "logs/$RUN.attempts"
  python scripts/train_ssm.py "${args[@]}" 2>&1 | tee -a "logs/train_$RUN.log"
  rc=${PIPESTATUS[0]}
  echo "== $(date '+%F %T') attempt $attempt exit $rc" | tee -a "logs/$RUN.attempts"
  [ "$rc" -eq 0 ] && { echo "run complete"; exit 0; }
  grep -h -E "OutOfMemoryError|RuntimeError|Error executing" "logs/train_$RUN.log" | tail -1 | tee -a "logs/$RUN.attempts"
  attempt=$((attempt + 1))
  [ "$attempt" -gt "$MAX_RETRIES" ] && { echo "MAX_RETRIES reached, STOP"; exit 1; }
  sleep 30
done
