#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
mkdir -p logs/final_u3

# These are the three variants not complete at repository commit 4451575.
for v in wide_d320_b4_r2 wide_deep_d320_b6_r2 mlpwide_d256_b4_r4; do
  sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" --job-name="u3-${v:0:18}" "$SCRIPT_DIR/upair_variant_24h.sbatch" "$v"
done

# Submit this only once; it produces the three common benchmark curves.
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" "$SCRIPT_DIR/baselines_24h.sbatch"
