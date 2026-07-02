from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import optuna
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from optuna_1dmrs_common import make_pruner, score_validation_row, validation_history_score  # noqa: E402
from upair5g.training import train_model  # noqa: E402


def _release_worker_memory() -> None:
    async_wait = getattr(getattr(tf, "experimental", object()), "async_wait", None)
    if callable(async_wait):
        try:
            async_wait()
        except Exception:
            pass
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


def _validation_callback_for_trial(args: argparse.Namespace, trial_ref: optuna.Trial | None):
    recent_values: list[float] = []

    def _callback(row: dict[str, Any], report_step: int) -> None:
        value = score_validation_row(row, args.objective_metric)
        recent_values.append(float(value))
        if args.objective_aggregation == "recent_mean":
            k = max(1, min(int(args.objective_recent_k), len(recent_values)))
            report_value = float(sum(recent_values[-k:]) / k)
        elif args.objective_aggregation == "best":
            report_value = float(min(recent_values))
        else:
            report_value = float(value)
        print(f"[OPTUNA-WORKER] trial={args.trial_number} report step={report_step} value={report_value:.8g}")
        if trial_ref is None:
            return
        try:
            trial_ref.report(float(report_value), int(report_step))
        except Exception as exc:
            print(f"[OPTUNA-WORKER] warning: could not report intermediate value: {exc!r}")
            return
        if int(report_step) < int(args.prune_warmup_steps):
            return
        try:
            should_prune = bool(trial_ref.should_prune())
        except Exception as exc:
            print(f"[OPTUNA-WORKER] warning: could not evaluate pruner: {exc!r}")
            return
        if should_prune:
            print(
                f"[OPTUNA-WORKER] trial={args.trial_number} pruned by {args.pruner} "
                f"at step={report_step} value={report_value:.8g}"
            )
            raise optuna.TrialPruned(f"pruned at step={report_step} value={report_value:.8g}")

    return _callback


def _make_trial_ref(args: argparse.Namespace) -> optuna.Trial | None:
    if args.disable_pruning or args.pruner == "none" or int(args.trial_id) < 0:
        return None
    try:
        storage = optuna.storages.RDBStorage(
            url=args.storage,
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
        study = optuna.load_study(study_name=args.study_name, storage=storage, pruner=make_pruner(args))
        return optuna.trial.Trial(study, int(args.trial_id))
    except Exception as exc:
        print(f"[OPTUNA-WORKER] warning: could not attach live Optuna trial for pruning: {exc!r}")
        return None


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-trial TensorFlow worker for isolated clean 1-DMRS Optuna.")
    parser.add_argument("--trial-config", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--trial-id", type=int, default=-1)
    parser.add_argument("--stage", default="A", choices=["A", "B", "C", "a", "b", "c"])
    parser.add_argument("--objective-metric", choices=["prop_nmse", "prop_nmse_ratio", "hybrid_nmse_ratio"], default="hybrid_nmse_ratio")
    parser.add_argument("--objective-aggregation", choices=["recent_mean", "last", "best"], default="recent_mean")
    parser.add_argument("--objective-recent-k", type=int, default=2)
    parser.add_argument("--objective-min-step", type=int, default=0)
    parser.add_argument("--pruner", choices=["percentile", "median", "successive_halving", "none"], default="percentile")
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--pruner-percentile", type=float, default=25.0)
    parser.add_argument("--pruner-startup-trials", type=int, default=10)
    parser.add_argument("--pruner-min-trials", type=int, default=5)
    parser.add_argument("--prune-warmup-steps", type=int, default=2000)
    parser.add_argument("--prune-interval-steps", type=int, default=1000)
    args = parser.parse_args()
    args.stage = str(args.stage).upper()

    result_path = Path(args.result_json)
    try:
        with open(args.trial_config, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)

        allocator = os.environ.get("TF_GPU_ALLOCATOR", "<unset/default>")
        print(
            f"[OPTUNA-WORKER] start trial={args.trial_number} pid={os.getpid()} "
            f"TF_GPU_ALLOCATOR={allocator} TF_FORCE_GPU_ALLOW_GROWTH={os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', '<unset>')}"
        )
        try:
            gpus = tf.config.list_physical_devices("GPU")
            print(f"[OPTUNA-WORKER] visible_gpus={gpus}")
        except Exception as exc:
            print(f"[OPTUNA-WORKER] gpu query failed: {exc!r}")

        trial_ref = _make_trial_ref(args)
        result = train_model(cfg, validation_callback=_validation_callback_for_trial(args, trial_ref))
        if not bool(result.get("training_complete", True)):
            _write_result(
                result_path,
                {
                    "status": "incomplete",
                    "trial_number": int(args.trial_number),
                    "latest_step": int(result.get("latest_step", -1)),
                    "total_steps": int(result.get("total_steps", -1)),
                    "history_path": result.get("history_path"),
                    "train_state_path": result.get("train_state_path"),
                },
            )
            print(f"[OPTUNA-WORKER] incomplete trial={args.trial_number}; leaving Optuna trial RUNNING for resume")
            return 143

        value = validation_history_score(
            Path(result["history_path"]),
            objective_metric=args.objective_metric,
            aggregation=args.objective_aggregation,
            recent_k=int(args.objective_recent_k),
            min_step=int(args.objective_min_step),
        )
        _write_result(
            result_path,
            {
                "status": "complete",
                "trial_number": int(args.trial_number),
                "value": float(value),
                "history_path": result.get("history_path"),
                "train_state_path": result.get("train_state_path"),
                "latest_step": int(result.get("latest_step", -1)),
                "total_steps": int(result.get("total_steps", -1)),
            },
        )
        print(f"[OPTUNA-WORKER] complete trial={args.trial_number} value={value:.8g}")
        return 0
    except optuna.TrialPruned as exc:
        _write_result(
            result_path,
            {
                "status": "pruned",
                "trial_number": int(args.trial_number),
                "message": str(exc),
            },
        )
        print(f"[OPTUNA-WORKER] pruned trial={args.trial_number}: {exc}")
        return 0
    except tf.errors.ResourceExhaustedError as exc:
        _release_worker_memory()
        _write_result(
            result_path,
            {
                "status": "resource_exhausted",
                "trial_number": int(args.trial_number),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"[OPTUNA-WORKER] resource_exhausted trial={args.trial_number}: {exc}")
        return 75
    except KeyboardInterrupt:
        _write_result(
            result_path,
            {
                "status": "interrupted",
                "trial_number": int(args.trial_number),
                "message": "KeyboardInterrupt or forwarded Slurm signal",
            },
        )
        return 143
    except Exception as exc:
        _release_worker_memory()
        _write_result(
            result_path,
            {
                "status": "failed",
                "trial_number": int(args.trial_number),
                "message": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"[OPTUNA-WORKER] failed trial={args.trial_number}: {exc!r}")
        traceback.print_exc()
        return 1
    finally:
        _release_worker_memory()


if __name__ == "__main__":
    raise SystemExit(main())
