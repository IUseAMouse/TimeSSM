#!/usr/bin/env bash
# Run train_patched.py until it reaches its last step, resuming from the run's checkpoint
# after a crash. Gives up after MAX_TRIES failures in a row without progress.
#
#   scripts/train_patched_loop.sh <config.yaml> run.name=<name> [overrides...]
set -u
CONFIG=$1; shift
NAME=$(printf '%s\n' "$@" | sed -n 's/^run\.name=//p' | tail -1)
OUT=$(printf '%s\n' "$@" | sed -n 's/^run\.out=//p' | tail -1)
OUT=${OUT:-$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').run.out)")}
[ -n "$NAME" ] || { echo "usage: $0 <config> run.name=<name> [overrides...]" >&2; exit 2; }
CKPT="$OUT/$NAME/$NAME.ckpt"
MAX_TRIES=${MAX_TRIES:-5}
step_of() { [ -f "$CKPT" ] && python -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False)['step'])" || echo 0; }
fails=0; last=$(step_of)
while true; do
  resume=(); [ -f "$CKPT" ] && resume=("run.resume=$CKPT")
  echo "$(date -u +%FT%TZ) start at step $last" >&2
  python "$(dirname "$0")/train_patched.py" --config "$CONFIG" "$@" "${resume[@]}" && exit 0
  now=$(step_of)
  if [ "$now" -gt "$last" ]; then fails=0; else fails=$((fails + 1)); fi
  last=$now
  [ "$fails" -ge "$MAX_TRIES" ] && { echo "no progress after $MAX_TRIES tries, stopping" >&2; exit 1; }
  sleep 60
done
