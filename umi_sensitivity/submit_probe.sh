#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate
python "$ROOT/umi_sensitivity/driver.py" static
mkdir -p "$ROOT/logs/umi_sensitivity"
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" "$ROOT/umi_sensitivity/probe_2h.sbatch"
