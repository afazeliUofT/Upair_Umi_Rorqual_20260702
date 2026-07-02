#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/rsadve1/links/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI"
cd "$ROOT"

OUT="UMI_training/diagnostics_rorqual"
mkdir -p "$OUT"

export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"

{
    echo "HOST=$(hostname)"
    echo "PWD=$PWD"
    echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "GIT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || true)"
    echo "PYTHON=$(which python)"
    python - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
print("python_executable =", sys.executable)
for p in ["numpy", "pandas", "tensorflow", "sionna-no-rt", "optuna"]:
    try:
        print(f"{p} = {version(p)}")
    except PackageNotFoundError:
        print(f"{p} = MISSING")
PY
} | tee "$OUT/env.txt"

echo
echo "========== CURRENT QUEUE =========="
squeue -u "$USER" | tee "$OUT/squeue.txt" || true

echo
echo "========== STAGE B STATUS =========="
python umi_training/driver.py study-status B | tee "$OUT/stageB_status.txt" || true

echo
echo "========== RECENT SACCT =========="
if command -v sacct >/dev/null 2>&1; then
    start_date="$(date -d '10 days ago' +%F 2>/dev/null || date +%F)"
    sacct -u "$USER" -S "$start_date" \
      --format=JobID%22,JobName%28,ArrayTaskID,State,ExitCode,Elapsed,NodeList,ReqTRES%45,MaxRSS,Reason \
      > "$OUT/sacct_recent.txt" || true
    tail -n 120 "$OUT/sacct_recent.txt"
else
    echo "sacct not available" | tee "$OUT/sacct_recent.txt"
fi

cat > "$OUT/audit_stageB.py" <<'PY'
from pathlib import Path
import json
import math
import re
from datetime import timezone

import optuna
import pandas as pd
from optuna.trial import TrialState

ROOT = Path.cwd()
OUT = ROOT / "UMI_training" / "diagnostics_rorqual"

VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]

PREFIX = "umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"
DB_ROOT = ROOT / "UMI_training" / "optuna_db"
RUN_ROOT = ROOT / "optuna" / "runs_1dmrs"
TARGET = 6


def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_failure(text: str) -> str:
    s = text.lower()

    if not s.strip():
        return ""

    if "cuda_error_not_initialized" in s or "tf_gpus = []" in s or "gpucheck-fatal" in s:
        return "gpucheck_no_tensorflow_gpu"

    if "cudasetdevice" in s or "context is destroyed" in s or "present state" in s:
        return "cuda_context_initialization"

    if (
        "resourceexhausted" in s
        or "cuda_error_out_of_memory" in s
        or "cumemallocasync failed" in s
        or "out of memory" in s
    ):
        return "true_resource_exhausted_oom"

    if "modulenotfounderror" in s or "no module named" in s:
        return "python_environment_missing_package"

    if "time limit" in s or "cancelled" in s:
        return "walltime_or_cancelled"

    if "traceback" in s or "failed trial" in s:
        return "other_python_exception"

    return "unknown"


def iso(dt):
    if dt is None:
        return ""
    try:
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return str(dt)


trial_rows = []
summary_rows = []

for variant in VARIANTS:
    study_name = f"{PREFIX}_{variant}"
    db = DB_ROOT / f"{study_name}.db"
    best_json = ROOT / "optuna" / f"{study_name}_best_params.json"

    if not db.exists():
        summary_rows.append({
            "variant": variant,
            "db_exists": False,
            "complete": 0,
            "pruned": 0,
            "failed": 0,
            "running": 0,
            "waiting": 0,
            "finished": 0,
            "target": TARGET,
            "ready": False,
            "best_value": math.nan,
            "best_trial": None,
            "needed_finished": TARGET,
        })
        continue

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db.resolve()}",
    )

    trials = study.get_trials(deepcopy=True)
    counts = {state.name: 0 for state in TrialState}
    for t in trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1

    complete_trials = [t for t in trials if t.state == TrialState.COMPLETE and t.value is not None]
    finished = counts.get("COMPLETE", 0) + counts.get("PRUNED", 0)
    active = counts.get("RUNNING", 0) + counts.get("WAITING", 0)
    ready = finished >= TARGET and counts.get("COMPLETE", 0) >= 1 and best_json.exists()

    best_value = min([float(t.value) for t in complete_trials], default=math.nan)
    best_trial = None
    if complete_trials:
        best_trial = min(complete_trials, key=lambda t: float(t.value)).number

    for t in trials:
        trial_dir = RUN_ROOT / study_name / f"{variant}_trial_{int(t.number):04d}"
        worker = read_json(trial_dir / "metrics" / "worker_result.json")
        train_state = read_json(trial_dir / "metrics" / "train_state.json")
        trial_cfg = read_json(trial_dir / "artifacts" / "trial_config.json")

        failure_text = " ".join([
            str(worker.get("status", "")),
            str(worker.get("message", "")),
            str(worker.get("traceback", "")),
        ])

        trial_rows.append({
            "variant": variant,
            "study_name": study_name,
            "trial": int(t.number),
            "state": t.state.name,
            "value": float(t.value) if t.value is not None else math.nan,
            "failure_class": classify_failure(failure_text) if t.state == TrialState.FAIL else "",
            "worker_status": worker.get("status", ""),
            "worker_message_short": str(worker.get("message", ""))[:240],
            "latest_step": train_state.get("latest_step", worker.get("latest_step", "")),
            "total_steps": train_state.get("total_steps", worker.get("total_steps", "")),
            "training_complete": train_state.get("training_complete", ""),
            "batch_size_train": trial_cfg.get("system", {}).get("batch_size_train", ""),
            "batch_size_eval": trial_cfg.get("system", {}).get("batch_size_eval", ""),
            "val_microbatch_size": trial_cfg.get("training", {}).get("val_microbatch_size", ""),
            "memory_cleanup_every_steps": trial_cfg.get("training", {}).get("memory_cleanup_every_steps", ""),
            "datetime_start": iso(t.datetime_start),
            "datetime_complete": iso(t.datetime_complete),
            **{f"param_{k}": v for k, v in t.params.items()},
        })

    summary_rows.append({
        "variant": variant,
        "db_exists": True,
        "complete": counts.get("COMPLETE", 0),
        "pruned": counts.get("PRUNED", 0),
        "failed": counts.get("FAIL", 0),
        "running": counts.get("RUNNING", 0),
        "waiting": counts.get("WAITING", 0),
        "finished": finished,
        "target": TARGET,
        "ready": ready,
        "best_value": best_value,
        "best_trial": best_trial,
        "best_json_exists": best_json.exists(),
        "needed_finished": max(0, TARGET - finished - active),
    })

summary = pd.DataFrame(summary_rows)
trials = pd.DataFrame(trial_rows)

summary.to_csv(OUT / "stageB_summary.csv", index=False)
trials.to_csv(OUT / "stageB_trials.csv", index=False)

print("\n========== SUMMARY TABLE ==========")
print(summary.to_string(index=False))

print("\n========== FAILURE CLASSES ==========")
if trials.empty or "failure_class" not in trials:
    print("No trial rows.")
else:
    failures = trials[trials["state"].eq("FAIL")].copy()
    if failures.empty:
        print("No failed Optuna trials.")
    else:
        print(
            failures.groupby(["variant", "failure_class"])
            .size()
            .reset_index(name="count")
            .to_string(index=False)
        )

print("\n========== BEST COMPLETED TRIALS ==========")
best_rows = []
for variant in VARIANTS:
    sub = trials[(trials["variant"].eq(variant)) & (trials["state"].eq("COMPLETE"))].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("value", ascending=True)
    best_rows.append(sub.iloc[0])
if best_rows:
    best = pd.DataFrame(best_rows)
    cols = [
        "variant", "trial", "value", "batch_size_train",
        "batch_size_eval", "val_microbatch_size",
        "memory_cleanup_every_steps",
        "param_learning_rate_schedule", "param_learning_rate",
        "param_weight_decay", "param_nmse_loss_weight",
        "param_dropout", "param_residual_scale",
    ]
    cols = [c for c in cols if c in best.columns]
    print(best[cols].to_string(index=False))
else:
    print("No completed trials.")

all_ready = bool(summary["ready"].all()) if not summary.empty else False
print(f"\nSTAGE_B_READY_BY_AUDIT={int(all_ready)}")
PY

python "$OUT/audit_stageB.py" | tee "$OUT/stageB_audit_report.txt"

cat > "$OUT/scan_stageB_logs.py" <<'PY'
from pathlib import Path
import re
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "UMI_training" / "diagnostics_rorqual"
LOG_ROOT = ROOT / "logs" / "umi_training"

patterns = {
    "gpucheck_pass": r"GPUCHECK\] PASS",
    "gpucheck_fail": r"GPUCHECK-FATAL|tf_gpus = \[\]|CUDA_ERROR_NOT_INITIALIZED",
    "cuda_context": r"cudaSetDevice|context is destroyed|present state",
    "oom": r"ResourceExhausted|CUDA_ERROR_OUT_OF_MEMORY|cuMemAllocAsync failed|out of memory",
    "env_missing": r"ModuleNotFoundError|No module named",
    "failed_trial": r"FAILED trial|failed trial",
    "complete_trial": r"completed trial|complete trial",
}

rows = []
for path in sorted(LOG_ROOT.glob("umi-optB*.out")):
    text = path.read_text(errors="ignore")
    task = ""
    m = re.search(r"_(\d+)_(\d+)\.out$", path.name)
    if m:
        job_id, task = m.group(1), m.group(2)
    else:
        job_id = ""
    host = ""
    mh = re.search(r"\[GPUCHECK\] host=([^\s]+)", text)
    if mh:
        host = mh.group(1)
    rows.append({
        "log": str(path),
        "job_id": job_id,
        "task": task,
        "host": host,
        **{k: len(re.findall(v, text, flags=re.I)) for k, v in patterns.items()},
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "stageB_log_scan.csv", index=False)

print("\n========== LOG SCAN ==========")
if df.empty:
    print("No umi-optB logs found.")
else:
    show = df[
        (df["gpucheck_fail"] > 0)
        | (df["cuda_context"] > 0)
        | (df["oom"] > 0)
        | (df["env_missing"] > 0)
        | (df["failed_trial"] > 0)
    ].copy()
    if show.empty:
        print("No suspicious Stage-B log patterns found.")
    else:
        print(show.to_string(index=False))
PY

python "$OUT/scan_stageB_logs.py" | tee "$OUT/stageB_log_scan_report.txt"

tar -czf UMI_training/stageB_rorqual_diagnostics.tar.gz \
  UMI_training/diagnostics_rorqual

echo
echo "[DONE] Diagnostic bundle:"
echo "UMI_training/stageB_rorqual_diagnostics.tar.gz"
