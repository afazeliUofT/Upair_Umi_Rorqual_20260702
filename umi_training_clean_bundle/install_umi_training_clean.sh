#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI}"
ROOT="$(cd "$ROOT" && pwd)"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$ROOT" == *"_underUMI" ]] || {
  echo "[FATAL] Expected copied _underUMI repository: $ROOT" >&2
  exit 2
}

required=(
  "$ROOT/upair_portable_env.sh"
  "$ROOT/src/upair5g/umi_channel.py"
  "$ROOT/configs/twc_comprehensive_mu32_umi_sensitivity.yaml"
  "$ROOT/umi_sensitivity/driver.py"
  "$ROOT/umi_sensitivity/PROBE_PASSED.json"
  "$ROOT/scripts/run_optuna_1dmrs_structure.py"
  "$ROOT/scripts/run_isolated_eval_chunk.py"
  "$BUNDLE_DIR/umi_training/driver.py"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "[FATAL] Missing: $path" >&2; exit 2; }
done

cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

mkdir -p \
  "$ROOT/UMI_training/optuna_db" \
  "$ROOT/logs/umi_training" \
  "$ROOT/umi_training"

cp -p "$BUNDLE_DIR"/umi_training/* "$ROOT/umi_training/"

python - <<'PYCFG'
from copy import deepcopy
from pathlib import Path
import yaml

root = Path.cwd()
src = root / "configs/twc_comprehensive_mu32_umi_sensitivity.yaml"
dst = root / "configs/twc_comprehensive_mu32_umi_training.yaml"
cfg = deepcopy(yaml.safe_load(src.read_text(encoding="utf-8")))
cfg["experiment"]["name"] = "rx16_prb8_1dmrs_umi_training"
cfg["experiment"]["training_domain"] = "umi"
cfg["experiment"]["channel_profile"] = "umi_standard_topology_normalized"
cfg["system"]["ebno_db_eval"] = [-4, -3, -2, -1, 0, 1]
cfg["training"]["steps"] = 40000
cfg["training"]["resume"] = True
cfg["training"]["checkpoint_every"] = 1000
cfg["training"]["eval_every"] = 2000
cfg["training"]["log_every"] = 100
cfg["training"]["val_steps"] = 96
cfg["training"]["val_ebno_db"] = [-4, -2, 0, 2, 4]
cfg["training"]["val_user_counts"] = [1, 2, 3, 4]
cfg["training"]["val_user_count_weights"] = [1.0, 3.0, 6.0, 10.0]
cfg["training"]["val_microbatch_size"] = 16
cfg["system"]["batch_size_train"] = 32
cfg["system"]["batch_size_eval"] = 32
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("[CONFIG] wrote", dst)
PYCFG

chmod +x "$ROOT"/umi_training/*.py "$ROOT"/umi_training/*.sh "$ROOT"/umi_training/*.sbatch
python -m py_compile "$ROOT/umi_training/driver.py"
for script in "$ROOT"/umi_training/*.sh "$ROOT"/umi_training/*.sbatch; do
  bash -n "$script"
done
python "$ROOT/umi_training/driver.py" static

echo "[OK] Installed readable UMi training workflow under: $ROOT/umi_training"
echo "[NEXT] bash umi_training/submit_smoke.sh"
