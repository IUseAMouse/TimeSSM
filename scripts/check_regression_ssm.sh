#!/usr/bin/env bash
# Backward-compatibility check on real weights: re-evaluate the published
# TimeSSM-2.5M checkpoint (epoch00_valloss1.2814, the one of the model card)
# with the current code, in a fresh directory, and compare each config's MASE
# and CRPS with cached results of the same checkpoint. Run it after any change
# to the model, the harness or the data path.
#   scripts/check_regression_ssm.sh            # ~10 min on one GPU
#   CONFIGS="ett1/H/short solar/10T/long" scripts/check_regression_ssm.sh
#   CK=<path to epoch00_valloss1.2814.ckpt> REF=<cached evaluation dir> scripts/check_regression_ssm.sh
# The bare model runs with the frequency given (ssm_mini_v3_freq_eval +freq_delta=true),
# the full model with every setting of the published number (timessm_2.5m_gift).
set -u
cd "$(dirname "$0")/.."
CK=${CK:-epoch00_valloss1.2814.ckpt}; STEM=$(basename "$CK" .ckpt)
REF=${REF:-evaluation/timessm_2.5m/$STEM}
NAME=_regression_check
CONFIGS=${CONFIGS:-"ett1/H/short m4_hourly/H/short solar/10T/long bizitobs_l2c/5T/long kdd_cup_2018/D/short"}
[ -f "$CK" ] && [ -d "$REF" ] || { echo "missing checkpoint ($CK) or cached results ($REF)"; exit 2; }
# Never reuse an earlier check's cache, never delete it either.
[ -d "evaluation/$NAME" ] && mv "evaluation/$NAME" "evaluation/${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
for c in $CONFIGS; do
  EVAL_CONFIG=ssm_mini_v3_freq_eval bash scripts/eval_ssm.sh "$CK" model.name=$NAME \
    "+gift_configs=$c" +gift_batch_size=32 +freq_delta=true >> logs/regression_bare.out 2>&1
  EVAL_CONFIG=timessm_2.5m_gift bash scripts/eval_ssm.sh "$CK" model.name=$NAME \
    "+gift_configs=$c" >> logs/regression_full.out 2>&1
done
python - "$REF" "evaluation/$NAME/$STEM" <<'PY'
import json, sys
from pathlib import Path
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
worst, n = 0.0, 0
for tag in sorted(d.name for d in new.iterdir() if (d / "per_config").is_dir()):
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
