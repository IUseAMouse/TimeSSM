#!/usr/bin/env bash
# Backward-compatibility check on real weights: re-evaluate the published
# 2.5M checkpoint (epoch00_valloss1.2814 of timessm_mini_v3_freq_zs) with the
# current code, in a fresh directory, in its two published settings: bare
# with the frequency given (ssm_mini_v3_freq_eval +freq_delta=true, model
# card MASE 0.7985 / CRPS 0.5451 on 97 configs) and the released inference
# stack (timessm_2.5m_gift, MASE 0.7544 / CRPS 0.5128). Each config's MASE
# and CRPS must equal the cached results. Run it after any change to the
# model, the harness or the data path.
#   scripts/check_regression_ssm.sh            # ~10 min on one GPU
#   CONFIGS="ett1/H/short solar/10T/long" scripts/check_regression_ssm.sh
#   CKPT=epoch00_valloss1.2814.ckpt scripts/check_regression_ssm.sh   # the file downloaded from Hugging Face
set -u
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints/timessm_mini_v3_freq_zs/pretrain_False/epoch00_valloss1.2814.ckpt}
CK=$(basename "$CKPT" .ckpt)
REF=evaluation/timessm_mini_v3_freq_zs/$CK
NAME=_regression_check
CONFIGS=${CONFIGS:-"ett1/H/short m4_hourly/H/short solar/10T/long bizitobs_l2c/5T/long kdd_cup_2018/D/short"}
BARE_TAG=gift_fdelta
STACK_TAG=gift_flip_ratein-mix-pool-up234-bt4-w4_fdelta_gamma-gamma_epoch00_valloss1.2814_flip_fdelta
[ -f "$CKPT" ] && [ -d "$REF/$BARE_TAG/per_config" ] && [ -d "$REF/$STACK_TAG/per_config" ] \
  || { echo "missing checkpoint or cached results"; exit 2; }
# Never reuse an earlier check's cache, never delete it either.
[ -d "evaluation/$NAME" ] && mv "evaluation/$NAME" "evaluation/${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
for c in $CONFIGS; do
  EVAL_CONFIG=ssm_mini_v3_freq_eval bash scripts/eval_ssm.sh "$CKPT" model.name=$NAME \
    "+gift_configs=$c" +gift_batch_size=32 +freq_delta=true >> logs/regression_nu.out 2>&1
  EVAL_CONFIG=timessm_2.5m_gift bash scripts/eval_ssm.sh "$CKPT" model.name=$NAME \
    "+gift_configs=$c" >> logs/regression_stack.out 2>&1
done
python - "$REF" "evaluation/$NAME/$CK" "$BARE_TAG" "$STACK_TAG" <<'PY'
import json, sys
from pathlib import Path
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
worst, n = 0.0, 0
for tag in sys.argv[3:]:
    for f in sorted((new / tag / "per_config").glob("*.json")):
        cached = ref / tag / "per_config" / f.name
        if not cached.exists():
            print(f"{tag}/{f.stem}: no cached result to compare with"); continue
        a, b = json.loads(f.read_text())["model"], json.loads(cached.read_text())["model"]
        d = max(abs(a[k] / b[k] - 1.0) for k in ("MASE", "CRPS"))
        worst, n = max(worst, d), n + 1
        print(f"{tag[:28]:28s} {f.stem:28s} MASE {a['MASE']:.6f} vs {b['MASE']:.6f} | "
              f"CRPS {a['CRPS']:.6f} vs {b['CRPS']:.6f} | {'identical' if d == 0 else f'rel. diff {d:.2e}'}")
ok = n > 0 and worst <= 1e-6
print(f"{n} comparisons, worst relative difference {worst:.2e}: " + ("OK" if ok else "REGRESSION"))
sys.exit(0 if ok else 1)
PY
