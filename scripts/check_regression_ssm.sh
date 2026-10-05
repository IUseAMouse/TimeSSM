#!/usr/bin/env bash
# Backward-compatibility check on REAL weights (2026-10-05): re-evaluate the
# published 2.5M checkpoint with the current code and no new flag, in a fresh
# directory, and compare each config's MASE and CRPS with the cached results
# of the model card. Run it before any arm that touched the model, the
# harness or the data path.
#   scripts/check_regression_ssm.sh            # ~10 min on one GPU
#   CONFIGS="ett1/H/short solar/10T/long" scripts/check_regression_ssm.sh
set -u
cd "$(dirname "$0")/.."
D=checkpoints/timessm_mini_v3_wide_zs/pretrain_False; CK=epoch00_valloss1.2841
REF=evaluation/timessm_mini_v3_zs/$CK
NAME=_regression_check
CONFIGS=${CONFIGS:-"ett1/H/short m4_hourly/H/short solar/10T/long bizitobs_l2c/5T/long kdd_cup_2018/D/short"}
STACK="+tta_flip=true +ratein=mix +ratein_pool=true"
[ -f "$D/$CK.ckpt" ] && [ -d "$REF/gift/per_config" ] || { echo "missing checkpoint or cached results"; exit 2; }
# never reuse an earlier check's cache, never delete it either
[ -d "evaluation/$NAME" ] && mv "evaluation/$NAME" "evaluation/${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
for c in $CONFIGS; do
  EVAL_CONFIG=ssm_mini_v3_wide_eval bash scripts/eval_ssm.sh "$D/$CK.ckpt" model.name=$NAME \
    "+gift_configs=$c" +gift_batch_size=32 >> logs/regression_nu.out 2>&1
  # shellcheck disable=SC2086
  EVAL_CONFIG=ssm_mini_v3_wide_eval bash scripts/eval_ssm.sh "$D/$CK.ckpt" model.name=$NAME \
    "+gift_configs=$c" +gift_batch_size=32 $STACK >> logs/regression_stack.out 2>&1
done
python - "$REF" "evaluation/$NAME/$CK" <<'PY'
import json, sys
from pathlib import Path
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
worst, n = 0.0, 0
for tag in ("gift", "gift_flip_ratein-mix-pool"):
    for f in sorted((new / tag / "per_config").glob("*.json")):
        cached = ref / tag / "per_config" / f.name
        if not cached.exists():
            print(f"{tag}/{f.stem}: no cached result to compare with"); continue
        a, b = json.loads(f.read_text())["model"], json.loads(cached.read_text())["model"]
        d = max(abs(a[k] / b[k] - 1.0) for k in ("MASE", "CRPS"))
        worst, n = max(worst, d), n + 1
        print(f"{tag:28s} {f.stem:28s} MASE {a['MASE']:.6f} vs {b['MASE']:.6f} | "
              f"CRPS {a['CRPS']:.6f} vs {b['CRPS']:.6f} | {'identical' if d == 0 else f'rel. diff {d:.2e}'}")
ok = n > 0 and worst <= 1e-6
print(f"{n} comparisons, worst relative difference {worst:.2e}: " + ("OK" if ok else "REGRESSION"))
sys.exit(0 if ok else 1)
PY
