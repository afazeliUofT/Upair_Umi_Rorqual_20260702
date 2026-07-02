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
