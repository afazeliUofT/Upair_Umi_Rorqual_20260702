#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

fail=0
ok(){ echo "[OK] $*"; }
bad(){ echo "[FAIL] $*" >&2; fail=1; }

for f in \
  upair_variant_pipeline_worker.sh \
  upair_submit_7variant_pipeline.sh \
  scripts/run_isolated_eval_chunk.py \
  scripts/merge_isolated_eval_chunks.py \
  scripts/isolated_eval_status.py \
  upair_portable_env.sh \
  upair_submit_lib.sh
do
  [[ -e "$f" ]] && ok "$f exists" || bad "$f missing"
done

grep -q "UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1" upair_variant_pipeline_worker.sh && ok "worker uses training-only mode before isolated eval" || bad "worker does not force training-only mode"
grep -q "run_isolated_eval_chunk.py" upair_variant_pipeline_worker.sh && ok "worker uses isolated eval chunks" || bad "worker does not use isolated eval"
grep -q "isolated_eval_status.py" upair_variant_pipeline_worker.sh && ok "worker has chunk-resume status logic" || bad "worker missing chunk-resume status logic"
grep -q "gpu:h100:1" upair_submit_7variant_pipeline.sh && ok "submit wrapper defaults to one H100 per variant" || bad "submit wrapper does not default to H100"

python - <<'PY'
from pathlib import Path
import ast, json, sys
for path in ["scripts/isolated_eval_status.py", "scripts/run_isolated_eval_chunk.py", "scripts/merge_isolated_eval_chunks.py"]:
    ast.parse(Path(path).read_text())
    print(f"[OK] Python syntax: {path}")

variants = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]
print("\n[TRAINING STATUS]")
for v in variants:
    p = Path(f"TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/{v}/metrics/train_state.json")
    if not p.exists():
        print(f"[MISSING] {v}: no train_state.json")
        continue
    d = json.loads(p.read_text())
    print(f"[{'COMPLETE' if d.get('training_complete') else 'INCOMPLETE'}] {v}: latest={d.get('latest_step')}/{d.get('total_steps')} reason={d.get('save_reason')} best_val={d.get('best_val')}")
PY

source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv

for v in $(source "${ROOT}/upair_submit_lib.sh" >/dev/null 2>&1; upair_variants); do
  if [[ -s "optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_${v}_best_params.json" || -s "optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB_${v}.db" ]]; then
    ok "Stage-B best exists for ${v}"
  else
    bad "Stage-B best missing for ${v}"
  fi
done

[[ "$fail" == "0" ]] || exit 1
echo "[PROBE] PASSED 7-variant master pipeline readiness probe"
