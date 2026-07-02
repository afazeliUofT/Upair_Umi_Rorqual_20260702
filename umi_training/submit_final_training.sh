#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate
python umi_training/driver.py study-status B
pending="$(python - <<'PY'
import json
from pathlib import Path
variants = [
    "main_d256_b4_r2","shallow_d256_b2_r2","deep_d256_b6_r2",
    "narrow_d192_b4_r2","wide_d320_b4_r2",
    "wide_deep_d320_b6_r2","mlpwide_d256_b4_r4",
]
root = Path("UMI_training/runs_rx16/seed7/1dmrs")
out = []
for i, variant in enumerate(variants):
    state = root / variant / "metrics/train_state.json"
    ckpt = root / variant / "checkpoints/best.weights.h5"
    complete = False
    if state.exists() and ckpt.exists():
        try:
            d = json.loads(state.read_text())
            complete = bool(d.get("training_complete", False)) and int(d.get("latest_step", -1)) == 40000
        except Exception:
            pass
    if not complete:
        out.append(str(i))
print(",".join(out))
PY
)"
if [[ -z "$pending" ]]; then
  echo "[FINAL-TRAIN] already complete"
  exit 0
fi
mkdir -p logs/umi_training
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" --array="${pending}%7" umi_training/final_train_array.sbatch
