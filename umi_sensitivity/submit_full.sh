#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

[[ -f "$ROOT/umi_sensitivity/PROBE_PASSED.json" ]] || {
  echo "[FATAL] Missing umi_sensitivity/PROBE_PASSED.json." >&2
  echo "        Run and inspect the mandatory GPU probe first." >&2
  exit 2
}
python "$ROOT/umi_sensitivity/driver.py" static
mkdir -p "$ROOT/logs/umi_sensitivity"

variants=(
  main_d256_b4_r2
  shallow_d256_b2_r2
  deep_d256_b6_r2
  narrow_d192_b4_r2
  wide_d320_b4_r2
  wide_deep_d320_b6_r2
  mlpwide_d256_b4_r4
)

for variant in "${variants[@]}"; do
  sbatch \
    --export=ALL,UPAIR_REPO_ROOT="$ROOT" \
    --job-name="umi-${variant:0:18}" \
    "$ROOT/umi_sensitivity/variant_24h.sbatch" "$variant"
done

sbatch \
  --export=ALL,UPAIR_REPO_ROOT="$ROOT" \
  "$ROOT/umi_sensitivity/baselines_24h.sbatch"
