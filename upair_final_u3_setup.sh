#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"

required=(
  upair_variant_pipeline_worker.sh
  upair_portable_env.sh
  scripts/run_isolated_eval_chunk.py
  scripts/isolated_eval_status.py
  scripts/merge_isolated_eval_chunks.py
  configs/twc_comprehensive_mu32_base.yaml
)
for p in "${required[@]}"; do
  [[ -e "$p" ]] || { echo "[FATAL] Missing $ROOT/$p" >&2; exit 2; }
done

mkdir -p final_u3_eval logs/final_u3 _final_u3_baseline_chunks _final_u3_shared_cov "temporary plots/final_u3"

cat > final_u3_eval/upair_variant_24h.sbatch <<'__VARIANT_SBATCH__'
#!/usr/bin/env bash
#SBATCH --job-name=u3-upair
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/final_u3/%x_%j.out

set -euo pipefail
VARIANT="${1:?Usage: sbatch final_u3_eval/upair_variant_24h.sbatch <variant>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

export UPAIR_DMRS_CASE="1dmrs"
export UPAIR_SEED="7"
export UPAIR_OPTUNA_STAGEB_PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
export UPAIR_EVAL_CHUNK_ROOT="$ROOT/_isolated_eval_chunks"
export UPAIR_PIPELINE_RECEIVERS="upair5g_lmmse"
export UPAIR_PIPELINE_USERS="3"
export UPAIR_PIPELINE_EBNOS="-4,-3,-2,-1,0,1"
export UPAIR_PIPELINE_CHUNK_BATCHES="20"
export UPAIR_PIPELINE_MICRO="8"
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS="100"
export UPAIR_PIPELINE_MAX_BATCHES="2000"
export UPAIR_PIPELINE_MIN_BATCHES="20"

signature() {
  python - "$VARIANT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
v = sys.argv[1]
state = Path(f"TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/{v}/metrics/train_state.json")
step = -1
complete = False
if state.exists():
    try:
        d = json.loads(state.read_text())
        step = int(d.get("latest_step", -1))
        complete = bool(d.get("training_complete", False))
    except Exception:
        pass
chunks = 0
root = Path("_isolated_eval_chunks")
for p in root.rglob("chunk_result.csv"):
    try:
        df = pd.read_csv(p, nrows=1)
        if df.empty:
            continue
        r = df.iloc[0]
        if (str(r.get("variant", "")) == v and
            str(r.get("receiver", "")) == "upair5g_lmmse" and
            int(r.get("num_users", -1)) == 3 and
            -4.0 <= float(r.get("ebno_db")) <= 1.0):
            chunks += 1
    except Exception:
        pass
print(f"step={step};complete={int(complete)};chunks={chunks}")
PY
}

last=""
stagnant=0
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[24H-WRAPPER] variant=$VARIANT attempt=$attempt"
  set +e
  bash "$ROOT/upair_variant_pipeline_worker.sh" "$VARIANT"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "[24H-WRAPPER] COMPLETE variant=$VARIANT"
    exit 0
  fi

  now="$(signature)"
  echo "[24H-WRAPPER] worker_rc=$rc progress=$now"
  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last="$now"

  if (( stagnant >= 2 )); then
    echo "[FATAL] No progress across three attempts; inspect this job log." >&2
    exit "$rc"
  fi
  sleep 15
done
__VARIANT_SBATCH__

cat > final_u3_eval/build_shared_covariance.py <<'__BUILD_COV__'
#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.run_comprehensive_mu32_ablation import (  # noqa: E402
    _apply_optuna_best_1dmrs,
    _eval_cfg,
    _variant_cfg,
)
from upair5g.baselines import estimate_empirical_covariances  # noqa: E402
from upair5g.builders import build_channel, build_pusch_transmitter  # noqa: E402
from upair5g.config import ensure_output_tree, load_config, set_cfg  # noqa: E402
from upair5g.utils import set_global_seed  # noqa: E402

VARIANT = "main_d256_b4_r2"
PREFIX = "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
SEED = 7
COV_SEED = 7007

base = load_config(ROOT / "configs" / "twc_comprehensive_mu32_base.yaml")
train_cfg = _variant_cfg(base, VARIANT, "1dmrs", SEED)
_apply_optuna_best_1dmrs(
    train_cfg,
    VARIANT,
    "1dmrs",
    storage_dir=ROOT / "optuna",
    study_prefix=PREFIX,
    require_external=True,
)
cfg = _eval_cfg(train_cfg, VARIANT, "1dmrs", 3)
set_cfg(cfg, "system.seed", COV_SEED)
set_cfg(cfg, "system.evaluation_seed", COV_SEED)
set_cfg(cfg, "multiuser.fixed_num_users", 3)
set_cfg(cfg, "experiment.output_root", str(ROOT / "_final_u3_shared_cov"))
set_cfg(cfg, "experiment.name", "u3_prb8_cdlC_covariance")
set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", True)
set_cfg(cfg, "baselines.covariance_estimation.cache_name", "empirical_covariances.npz")

set_global_seed(COV_SEED)
paths = ensure_output_tree(cfg)
tx, _ = build_pusch_transmitter(cfg, num_users=3)
channel = build_channel(cfg, tx)
result = estimate_empirical_covariances(tx=tx, channel=channel, cfg=cfg, paths=paths)
cache = Path(str(result["cache_path"].numpy().decode() if hasattr(result["cache_path"], "numpy") else result["cache_path"]))

manifest = {
    "cache": str(cache),
    "num_users": 3,
    "n_size_grid": int(cfg["pusch"]["n_size_grid"]),
    "channel_model": str(cfg["channel"]["model"]),
    "delay_spread_s": float(cfg["channel"]["delay_spread_s"]),
    "min_speed_mps": float(cfg["channel"]["min_speed_mps"]),
    "max_speed_mps": float(cfg["channel"]["max_speed_mps"]),
    "num_rx_ant": int(cfg["channel"]["num_rx_ant"]),
    "num_batches": int(cfg["baselines"]["covariance_estimation"]["num_batches"]),
    "batch_size": int(cfg["baselines"]["covariance_estimation"]["batch_size"]),
    "seed": COV_SEED,
}
(cache.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(cache)
__BUILD_COV__

cat > final_u3_eval/run_chunk_with_shared_cov.py <<'__SHARED_WRAPPER__'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def safe_tag(value: object) -> str:
    return str(value).replace("-", "m").replace("+", "p").replace(".", "p").replace(",", "_")


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--dmrs-case", default="1dmrs")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--num-users", type=int, required=True)
ap.add_argument("--receiver", required=True)
ap.add_argument("--ebno-db", type=float, required=True)
ap.add_argument("--chunk-idx", type=int, required=True)
ap.add_argument("--chunk-batches", type=int, default=20)
ap.add_argument("--receiver-microbatch-size", type=int, default=8)
ap.add_argument("--stageb-prefix", required=True)
ap.add_argument("--optuna-dir", required=True)
ap.add_argument("--output-root", required=True)
ap.add_argument("--shared-cov-cache", required=True)
args = ap.parse_args()

if args.receiver != "baseline_ls_2dlmmse_lmmse":
    raise SystemExit("This wrapper is only for baseline_ls_2dlmmse_lmmse")

shared = Path(args.shared_cov_cache).resolve()
if not shared.is_file():
    raise FileNotFoundError(shared)

with open(args.config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cache_name = str(cfg.get("baselines", {}).get("covariance_estimation", {}).get("cache_name", "empirical_covariances.npz"))

tag = (
    f"{args.variant}_u{args.num_users}_{args.receiver}_"
    f"ebno{safe_tag(args.ebno_db)}_chunk{args.chunk_idx:04d}_"
    f"m{args.receiver_microbatch_size}_b{args.chunk_batches}"
)
target = Path(args.output_root).resolve() / tag / "artifacts" / cache_name
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(shared, target)

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_isolated_eval_chunk.py"),
    "--config", args.config,
    "--variant", args.variant,
    "--dmrs-case", args.dmrs_case,
    "--seed", str(args.seed),
    "--num-users", str(args.num_users),
    "--receiver", args.receiver,
    "--ebno-db", str(args.ebno_db),
    "--chunk-idx", str(args.chunk_idx),
    "--chunk-batches", str(args.chunk_batches),
    "--receiver-microbatch-size", str(args.receiver_microbatch_size),
    "--stageb-prefix", args.stageb_prefix,
    "--optuna-dir", args.optuna_dir,
    "--output-root", args.output_root,
]
print("[SHARED-COV] staged", shared, "->", target)
subprocess.run(cmd, check=True, cwd=ROOT)
__SHARED_WRAPPER__

cat > final_u3_eval/baseline_worker.sh <<'__BASELINE_WORKER__'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="$ROOT/configs/twc_comprehensive_mu32_base.yaml"
VARIANT="main_d256_b4_r2"
PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
OUT_ROOT="$ROOT/_final_u3_baseline_chunks"
SHARED="$ROOT/_final_u3_shared_cov/u3_prb8_cdlC_covariance/artifacts/empirical_covariances.npz"
CHUNK_BATCHES=20
MICRO=8
TARGET=100
MAX_BATCHES=2000
MIN_BATCHES=20

python "$SCRIPT_DIR/build_shared_covariance.py"
[[ -s "$SHARED" ]] || { echo "[FATAL] Missing shared covariance: $SHARED" >&2; exit 3; }
mkdir -p "$OUT_ROOT"

receivers=(baseline_ls_lmmse baseline_ls_2dlmmse_lmmse perfect_csi_lmmse)
ebnos=(-4 -3 -2 -1 0 1)

for receiver in "${receivers[@]}"; do
  for ebno in "${ebnos[@]}"; do
    echo "================================================================================"
    echo "[FINAL-BASELINE] receiver=$receiver U=3 Eb/N0=$ebno"
    while true; do
      status_file="$(mktemp)"
      python "$ROOT/scripts/isolated_eval_status.py" \
        --input-root "$OUT_ROOT" \
        --config "$CONFIG" \
        --variant "$VARIANT" \
        --receiver "$receiver" \
        --num-users 3 \
        --ebno-db "$ebno" \
        --chunk-batches "$CHUNK_BATCHES" \
        --target-block-errors "$TARGET" \
        --max-batches "$MAX_BATCHES" \
        --min-batches "$MIN_BATCHES" \
        --shell > "$status_file"
      # shellcheck disable=SC1090
      source "$status_file"
      rm -f "$status_file"
      echo "[FINAL-BASELINE] done=$DONE reason=$REASON batches=$NUM_BATCHES errors=$BLOCK_ERRORS next=$NEXT_CHUNK"
      [[ "$DONE" == "1" ]] && break

      common=(
        --config "$CONFIG"
        --variant "$VARIANT"
        --dmrs-case 1dmrs
        --seed 7
        --num-users 3
        --receiver "$receiver"
        --ebno-db "$ebno"
        --chunk-idx "$NEXT_CHUNK"
        --chunk-batches "$CHUNK_BATCHES"
        --receiver-microbatch-size "$MICRO"
        --stageb-prefix "$PREFIX"
        --optuna-dir "$ROOT/optuna"
        --output-root "$OUT_ROOT"
      )

      if [[ "$receiver" == "baseline_ls_2dlmmse_lmmse" ]]; then
        python -u "$SCRIPT_DIR/run_chunk_with_shared_cov.py" \
          "${common[@]}" --shared-cov-cache "$SHARED"
      else
        python -u "$ROOT/scripts/run_isolated_eval_chunk.py" "${common[@]}"
      fi
    done

    safe="${ebno//-/m}"
    safe="${safe//./p}"
    python "$ROOT/scripts/merge_isolated_eval_chunks.py" \
      --input-root "$OUT_ROOT" \
      --output-csv "$OUT_ROOT/merged_${VARIANT}_u3_${receiver}_e${safe}.csv" \
      --variant "$VARIANT" \
      --receiver "$receiver" \
      --num-users 3 \
      --ebno-db "$ebno"
  done
done

echo "[FINAL-BASELINE] COMPLETE"
__BASELINE_WORKER__

cat > final_u3_eval/baselines_24h.sbatch <<'__BASELINE_SBATCH__'
#!/usr/bin/env bash
#SBATCH --job-name=u3-baselines
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/final_u3/%x_%j.out

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

last_count=-1
stagnant=0
while true; do
  set +e
  bash "$SCRIPT_DIR/baseline_worker.sh"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    exit 0
  fi
  count="$(find "$ROOT/_final_u3_baseline_chunks" -name chunk_result.csv -type f | wc -l)"
  echo "[24H-WRAPPER] baseline worker rc=$rc chunk_count=$count"
  if [[ "$count" == "$last_count" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last_count="$count"
  if (( stagnant >= 2 )); then
    echo "[FATAL] No baseline progress across three attempts." >&2
    exit "$rc"
  fi
  sleep 15
done
__BASELINE_SBATCH__

cat > final_u3_eval/check_status.py <<'__CHECK_STATUS__'
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPAIR_ROOT = ROOT / "_isolated_eval_chunks"
BASE_ROOT = ROOT / "_final_u3_baseline_chunks"
VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]
BASELINES = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "perfect_csi_lmmse",
]
EBNOS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]


def training_state(variant: str) -> tuple[str, int, int, str]:
    p = ROOT / "TWC_plots_comprehensive" / "runs_rx16" / "seed7" / "1dmrs" / variant / "metrics" / "train_state.json"
    if not p.exists():
        return "MISSING", -1, 40000, "no train_state.json"
    try:
        d = json.loads(p.read_text())
        return (
            "COMPLETE" if d.get("training_complete") else "INCOMPLETE",
            int(d.get("latest_step", -1)),
            int(d.get("total_steps", 40000)),
            str(d.get("save_reason", "")),
        )
    except Exception as exc:
        return "INVALID", -1, 40000, str(exc)


def point(root: Path, variant: str, receiver: str, ebno: float) -> tuple[int, int, bool]:
    frames = []
    for p in root.rglob("chunk_result.csv"):
        try:
            d = pd.read_csv(p)
            if d.empty:
                continue
            r = d.iloc[0]
            if (str(r.get("variant", "")) == variant and
                str(r.get("receiver", "")) == receiver and
                int(r.get("num_users", -1)) == 3 and
                abs(float(r.get("ebno_db")) - ebno) < 1e-9):
                frames.append(d)
        except Exception:
            pass
    if not frames:
        return 0, 0, False
    df = pd.concat(frames, ignore_index=True)
    errors = int(pd.to_numeric(df.get("block_errors"), errors="coerce").fillna(0).sum())
    batches = int(pd.to_numeric(df.get("num_batches_run"), errors="coerce").fillna(0).sum())
    done = (batches >= 20 and errors >= 100) or batches >= 2000
    return errors, batches, done


print("TRAINING")
all_training = True
for v in VARIANTS:
    status, step, total, reason = training_state(v)
    all_training &= status == "COMPLETE"
    print(f"{v:27s} {status:10s} {step:5d}/{total:<5d} reason={reason}")

print("\nUPAIR U=3, Eb/N0=-4..+1")
all_upair = True
for v in VARIANTS:
    states = []
    for e in EBNOS:
        err, batches, done = point(UPAIR_ROOT, v, "upair5g_lmmse", e)
        all_upair &= done
        states.append(f"{e:+g}:{'D' if done else 'M'}({err}/{batches})")
    print(f"{v:27s} " + "  ".join(states))

print("\nBASELINES U=3, Eb/N0=-4..+1")
all_base = True
for r in BASELINES:
    states = []
    for e in EBNOS:
        err, batches, done = point(BASE_ROOT, "main_d256_b4_r2", r, e)
        all_base &= done
        states.append(f"{e:+g}:{'D' if done else 'M'}({err}/{batches})")
    print(f"{r:35s} " + "  ".join(states))

print("\nLegend: D=done under 100-errors-or-2000-batches policy; M=missing/incomplete.")
print(f"OVERALL_COMPLETE={int(all_training and all_upair and all_base)}")
__CHECK_STATUS__

cat > final_u3_eval/plot_final.py <<'__PLOT_FINAL__'
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "temporary plots" / "final_u3"
OUT.mkdir(parents=True, exist_ok=True)
UPAIR_ROOT = ROOT / "_isolated_eval_chunks"
BASE_ROOT = ROOT / "_final_u3_baseline_chunks"
EBNOS = {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0}
VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]
BASELINES = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "perfect_csi_lmmse",
]
LABELS = {
    "main_d256_b4_r2": "UPAIR main (d=256,L=4,r=2)",
    "shallow_d256_b2_r2": "UPAIR shallow (d=256,L=2,r=2)",
    "deep_d256_b6_r2": "UPAIR deep (d=256,L=6,r=2)",
    "narrow_d192_b4_r2": "UPAIR narrow (d=192,L=4,r=2)",
    "wide_d320_b4_r2": "UPAIR wide (d=320,L=4,r=2)",
    "wide_deep_d320_b6_r2": "UPAIR wide-deep (d=320,L=6,r=2)",
    "mlpwide_d256_b4_r4": "UPAIR MLP-wide (d=256,L=4,r=4)",
    "baseline_ls_lmmse": "LS + LMMSE detector",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D-LMMSE + LMMSE detector",
    "perfect_csi_lmmse": "Perfect CSI + LMMSE detector",
}
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]


def load_rows(root: Path, receivers: set[str]) -> pd.DataFrame:
    frames = []
    for p in root.glob("merged_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        df["source"] = str(p.relative_to(ROOT))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ("num_users", "ebno_db", "bler", "block_errors", "num_blocks", "num_batches_run"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[
        df["receiver"].astype(str).isin(receivers)
        & df["num_users"].eq(3)
        & df["ebno_db"].isin(EBNOS)
    ].copy()
    df = df.sort_values("num_blocks").drop_duplicates(
        ["variant", "receiver", "num_users", "ebno_db"], keep="last"
    )
    return df


upair = load_rows(UPAIR_ROOT, {"upair5g_lmmse"})
upair = upair[upair["variant"].isin(VARIANTS)]
baselines = load_rows(BASE_ROOT, set(BASELINES))
baselines = baselines[baselines["variant"].eq("main_d256_b4_r2")]
all_rows = pd.concat([upair, baselines], ignore_index=True, sort=False)
all_rows["reliable_bler"] = all_rows["block_errors"].fillna(0).ge(100)
all_rows["done"] = (
    (all_rows["num_batches_run"].fillna(0).ge(20) & all_rows["block_errors"].fillna(0).ge(100))
    | all_rows["num_batches_run"].fillna(0).ge(2000)
)
all_rows.to_csv(OUT / "final_u3_bler_rows.csv", index=False)

expected = []
for v in VARIANTS:
    for e in sorted(EBNOS):
        expected.append((v, "upair5g_lmmse", e))
for r in BASELINES:
    for e in sorted(EBNOS):
        expected.append(("main_d256_b4_r2", r, e))
observed = set(zip(all_rows["variant"], all_rows["receiver"], all_rows["ebno_db"]))
missing = pd.DataFrame(
    [(v, r, e) for v, r, e in expected if (v, r, e) not in observed],
    columns=["variant", "receiver", "ebno_db"],
)
missing.to_csv(OUT / "missing_points.csv", index=False)
all_rows[all_rows["bler"].fillna(0).le(0)].to_csv(OUT / "zero_bler_points.csv", index=False)


def plot(series_keys: list[tuple[str, str]], filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    for i, (variant, receiver) in enumerate(series_keys):
        s = all_rows[
            all_rows["variant"].eq(variant) & all_rows["receiver"].eq(receiver)
        ].sort_values("ebno_db")
        s = s[np.isfinite(s["bler"]) & s["bler"].gt(0)]
        if s.empty:
            continue
        key = variant if receiver == "upair5g_lmmse" else receiver
        linestyle = "-" if receiver == "upair5g_lmmse" else "--"
        line, = ax.plot(
            s["ebno_db"], s["bler"], linestyle=linestyle, linewidth=1.8,
            label=LABELS[key], zorder=2,
        )
        reliable = s["reliable_bler"].to_numpy(dtype=bool)
        x = s["ebno_db"].to_numpy(dtype=float)
        y = s["bler"].to_numpy(dtype=float)
        marker = MARKERS[i % len(MARKERS)]
        if reliable.any():
            ax.scatter(x[reliable], y[reliable], marker=marker, s=52,
                       color=line.get_color(), zorder=3)
        if (~reliable).any():
            ax.scatter(x[~reliable], y[~reliable], marker=marker, s=58,
                       facecolors="none", edgecolors=line.get_color(),
                       linewidths=1.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(sorted(EBNOS))
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("BLER")
    ax.set_title(title + "\n3 active users")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.text(
        0.01, 0.01,
        "Filled: ≥100 block errors; open: <100 errors at the 2000-batch cap. "
        "Zero/unavailable points are omitted.",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.2)
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)

upair_keys = [(v, "upair5g_lmmse") for v in VARIANTS]
base_keys = [("main_d256_b4_r2", r) for r in BASELINES]
plot(upair_keys + base_keys, "bler_all_7_upair_and_3_benchmarks_u3",
     "Extended UPAIR variants and benchmarks")
plot(upair_keys, "bler_all_7_upair_variants_u3", "Extended UPAIR architecture comparison")
plot([("main_d256_b4_r2", "upair5g_lmmse")] + base_keys,
     "bler_main_upair_vs_benchmarks_u3", "Main Extended UPAIR versus benchmarks")

print("[PLOT] rows:", len(all_rows))
print("[PLOT] missing points:", len(missing))
print("[PLOT] output:", OUT)
__PLOT_FINAL__

cat > final_u3_eval/submit.sh <<'__SUBMIT__'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
mkdir -p logs/final_u3

# These are the three variants not complete at repository commit 4451575.
for v in wide_d320_b4_r2 wide_deep_d320_b6_r2 mlpwide_d256_b4_r4; do
  sbatch --job-name="u3-${v:0:18}" "$SCRIPT_DIR/upair_variant_24h.sbatch" "$v"
done

# Submit this only once; it produces the three common benchmark curves.
sbatch "$SCRIPT_DIR/baselines_24h.sbatch"
__SUBMIT__

chmod +x \
  final_u3_eval/build_shared_covariance.py \
  final_u3_eval/run_chunk_with_shared_cov.py \
  final_u3_eval/baseline_worker.sh \
  final_u3_eval/check_status.py \
  final_u3_eval/plot_final.py \
  final_u3_eval/submit.sh

echo "[OK] Installed final_u3_eval under: $ROOT/final_u3_eval"
echo "[NEXT] python final_u3_eval/check_status.py"
echo "[NEXT] bash final_u3_eval/submit.sh"
