#!/usr/bin/env bash
# Clean top-level redundant .sh files and transient runtime artifacts.
# Designed for the CDL-C Nibi repo after the master pipeline is installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

MODE="${UPAIR_CLEAN_MODE:-dryrun}"   # dryrun or apply
echo "[CLEAN] MODE=${MODE}"
echo "[CLEAN] ROOT=${ROOT}"

KEEP_SH=(
  upair_portable_env.sh
  upair_submit_lib.sh
  upair_variant_pipeline_worker.sh
  upair_submit_7variant_pipeline.sh
  upair_probe_pipeline_ready.sh
  upair_clean_top_level_sh_minimal.sh
)

is_keep() {
  local f="$1"
  for k in "${KEEP_SH[@]}"; do
    [[ "$f" == "$k" ]] && return 0
  done
  return 1
}

echo
echo "================================================================================"
echo "[CLEAN] Top-level .sh files"
mapfile -t all_sh < <(find . -maxdepth 1 -type f -name '*.sh' -printf '%f\n' | sort)
for f in "${all_sh[@]}"; do
  if is_keep "$f"; then
    echo "[KEEP]   $f"
  else
    echo "[REMOVE] $f"
  fi
done

echo
echo "================================================================================"
echo "[CLEAN] Transient folders/files to remove locally"
TRANSIENT=(
  _isolated_eval_chunks_smoke
  _stress_cdl_eval_memory
  _stress_cdl_eval_one_receiver
  _smoke_umi_norm_runtime
  _smoke_umi_pc_runtime
  _smoke_true_dmrs_runtime
  _smoke_*
  logs/eval_iso
  logs/train_eval
)

for x in "${TRANSIENT[@]}"; do
  compgen -G "$x" >/dev/null || continue
  for y in $x; do
    [[ -e "$y" ]] && echo "[REMOVE] $y"
  done
done

echo
echo "================================================================================"
echo "[CLEAN] Generated artifacts to untrack from git only"
echo "[UNTRACK] TWC_plots_comprehensive/  logs/  optuna/  _isolated_eval_chunks*/"

if [[ "$MODE" != "apply" ]]; then
  echo
  echo "[CLEAN] Dry run only. To apply:"
  echo "  UPAIR_CLEAN_MODE=apply bash upair_clean_top_level_sh_minimal.sh"
  exit 0
fi

echo
echo "[CLEAN] Applying cleanup..."

# Remove redundant top-level shell files
for f in "${all_sh[@]}"; do
  if ! is_keep "$f"; then
    rm -f "$f"
  fi
done

# Remove transient folders only. Do NOT remove TWC_plots_comprehensive/runs_rx16 or optuna from disk.
rm -rf _isolated_eval_chunks_smoke _stress_cdl_eval_memory _stress_cdl_eval_one_receiver _smoke_* logs/eval_iso logs/train_eval
mkdir -p logs/pipeline logs/submit

find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# Update .gitignore
python - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = [
    "TWC_plots_comprehensive/",
    "logs/",
    "optuna/",
    "_isolated_eval_chunks*/",
    "_stress_*/",
    "_smoke_*/",
    "__pycache__/",
    "*.py[cod]",
    "*.out",
    "*.err",
    "*.log",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.weights.h5",
    "*.data-*",
    "*.index",
    "checkpoint",
]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
PY

# Untrack generated artifacts only; keep files on disk.
git rm -r --cached --ignore-unmatch TWC_plots_comprehensive logs optuna _isolated_eval_chunks _isolated_eval_chunks_smoke _stress_cdl_eval_memory _stress_cdl_eval_one_receiver >/dev/null 2>&1 || true

echo
echo "[CLEAN] Final top-level .sh files:"
find . -maxdepth 1 -type f -name '*.sh' -printf '%f\n' | sort

echo
echo "[CLEAN] Done. Review:"
echo "  git status --short"
