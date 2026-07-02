from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from optuna_1dmrs_common import (  # noqa: E402
    ESSENTIAL_PARAM_NAMES,
    add_common_cli_args,
    apply_stage_defaults,
    candidate_already_present,
    completed_trial_count,
    enqueue_stage_candidates,
    export_best_params,
    failed_trial_count,
    finished_trial_count,
    has_resumable_training_state,
    history_path_for_trial,
    load_source_candidates,
    make_pruner,
    make_sampler,
    make_trial_config,
    nonfailed_trial_count,
    normalize_optuna_params,
    pruned_trial_count,
    read_train_state,
    suggest_trial_params,
    trial_output_root,
    validate_common_args,
    validation_history_score,
)

_CURRENT_CHILD: subprocess.Popen[Any] | None = None
_STOP_REQUESTED = False


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED, _CURRENT_CHILD
    _STOP_REQUESTED = True
    print(f"[OPTUNA-ISO] received signal {signum}; forwarding to current worker and leaving trial RUNNING.", flush=True)
    child = _CURRENT_CHILD
    if child is not None and child.poll() is None:
        try:
            child.terminate()
        except Exception:
            pass


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _signal_handler)
        except Exception:
            pass
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def _json_dump_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def _trial_id_from_trial_or_frozen(trial_obj: Any) -> int:
    for name in ("_trial_id", "trial_id"):
        value = getattr(trial_obj, name, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return -1


def _source_completed_candidates_for_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Thin wrapper to keep old import lints quiet and expose better error context.
    return load_source_candidates(args)


def _run_worker_for_trial(
    study: optuna.Study,
    args: argparse.Namespace,
    trial_ref: optuna.Trial | int,
    trial_number: int,
    params: dict[str, Any],
    trial_id: int = -1,
) -> bool:
    """Run exactly one TensorFlow training trial in a fresh Python process.

    Returning False means the worker stopped because of wall-time/signal.  In that case the Optuna
    trial is deliberately left RUNNING so a re-submitted job resumes the same trial directory.
    """
    global _CURRENT_CHILD, _STOP_REQUESTED

    params = normalize_optuna_params(args, params)
    trial_root = trial_output_root(args, trial_number)
    artifacts = trial_root / "artifacts"
    result_path = trial_root / "metrics" / "worker_result.json"
    trial_config_path = artifacts / "trial_config.json"
    params_path = artifacts / "optuna_params.json"

    cfg = make_trial_config(args, params, trial_number)
    _json_dump_atomic(trial_config_path, cfg)
    _json_dump_atomic(
        params_path,
        {
            "trial_number": int(trial_number),
            "trial_id": int(trial_id),
            "params": params,
            "stage": args.stage,
            "study_name": args.study_name,
        },
    )
    if result_path.exists():
        try:
            result_path.unlink()
        except Exception:
            pass

    worker = PROJECT_ROOT / "scripts" / "run_optuna_1dmrs_trial_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        "--trial-config",
        str(trial_config_path),
        "--result-json",
        str(result_path),
        "--study-name",
        str(args.study_name),
        "--storage",
        str(args.storage),
        "--trial-number",
        str(trial_number),
        "--trial-id",
        str(int(trial_id)),
        "--stage",
        str(args.stage),
        "--objective-metric",
        str(args.objective_metric),
        "--objective-aggregation",
        str(args.objective_aggregation),
        "--objective-recent-k",
        str(int(args.objective_recent_k)),
        "--objective-min-step",
        str(int(args.objective_min_step)),
        "--pruner",
        str(args.pruner),
        "--pruner-percentile",
        str(float(args.pruner_percentile)),
        "--pruner-startup-trials",
        str(int(args.pruner_startup_trials)),
        "--pruner-min-trials",
        str(int(args.pruner_min_trials)),
        "--prune-warmup-steps",
        str(int(args.prune_warmup_steps)),
        "--prune-interval-steps",
        str(int(args.prune_interval_steps)),
    ]
    if args.disable_pruning:
        cmd.append("--disable-pruning")

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("MPLBACKEND", "Agg")

    print(f"[OPTUNA-ISO] launching worker trial={trial_number} pid=<pending> params={params}", flush=True)
    _CURRENT_CHILD = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    returncode = _CURRENT_CHILD.wait()
    _CURRENT_CHILD = None

    if _STOP_REQUESTED or returncode == 143:
        print(f"[OPTUNA-ISO] worker interrupted trial={trial_number}; leaving Optuna state RUNNING.", flush=True)
        return False

    payload: dict[str, Any] = {}
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception as exc:
            print(f"[OPTUNA-ISO] could not read worker result {result_path}: {exc!r}", flush=True)

    status = str(payload.get("status", "missing_result"))
    if returncode == 0 and status == "complete":
        value = float(payload["value"])
        study.tell(trial_ref, value)
        print(f"[OPTUNA-ISO] completed trial={trial_number} value={value:.8g}", flush=True)
        export_best_params(study, args)
        return True
    if returncode == 0 and status == "pruned":
        study.tell(trial_ref, state=optuna.trial.TrialState.PRUNED)
        print(f"[OPTUNA-ISO] pruned trial={trial_number}: {payload.get('message', '')}", flush=True)
        export_best_params(study, args)
        return True
    if status in {"incomplete", "interrupted"}:
        print(f"[OPTUNA-ISO] incomplete/interrupted trial={trial_number}; leaving RUNNING for resume.", flush=True)
        return False
    if status == "resource_exhausted" or returncode == 75:
        # Resource exhaustion is not a valid Optuna pruning event.  Mark FAIL so it is not counted
        # toward Stage-A/B/C target trials and cannot silently poison the study statistics.
        try:
            setter = getattr(trial_ref, "set_user_attr", None)
            if callable(setter):
                setter("failure_kind", "resource_exhausted")
        except Exception:
            pass
        study.tell(trial_ref, state=optuna.trial.TrialState.FAIL)
        print(
            f"[OPTUNA-ISO] FAILED trial={trial_number} due to resource_exhausted; "
            "not counted as finished tuning evidence.",
            flush=True,
        )
        export_best_params(study, args)
        return True

    try:
        setter = getattr(trial_ref, "set_user_attr", None)
        if callable(setter):
            setter("failure_kind", status)
    except Exception:
        pass
    study.tell(trial_ref, state=optuna.trial.TrialState.FAIL)
    print(
        f"[OPTUNA-ISO] FAILED trial={trial_number} returncode={returncode} status={status}; "
        "not counted as finished tuning evidence.",
        flush=True,
    )
    export_best_params(study, args)
    return True


def _resume_running_trials(study: optuna.Study, args: argparse.Namespace) -> bool:
    made_progress = False
    running = [trial for trial in study.get_trials(deepcopy=False) if trial.state == optuna.trial.TrialState.RUNNING]
    for frozen in sorted(running, key=lambda t: t.number):
        params = normalize_optuna_params(args, dict(frozen.params))
        missing = [name for name in ESSENTIAL_PARAM_NAMES if name not in params]
        if missing:
            print(f"[OPTUNA-ISO] cannot resume trial={frozen.number}; missing params={missing}")
            continue
        state_payload = read_train_state(args, frozen.number)
        history_path = history_path_for_trial(args, frozen.number)
        if state_payload and bool(state_payload.get("optuna_pruned", False)):
            study.tell(frozen.number, state=optuna.trial.TrialState.PRUNED)
            print(f"[OPTUNA-ISO] finalized previously pruned RUNNING trial={frozen.number}")
            made_progress = True
            continue
        if state_payload and bool(state_payload.get("training_complete", False)) and history_path.exists():
            value = validation_history_score(
                history_path,
                objective_metric=args.objective_metric,
                aggregation=args.objective_aggregation,
                recent_k=int(args.objective_recent_k),
                min_step=int(args.objective_min_step),
            )
            study.tell(frozen.number, value)
            print(f"[OPTUNA-ISO] finalized completed RUNNING trial={frozen.number} value={value:.8g}")
            export_best_params(study, args)
            made_progress = True
            continue
        has_state = has_resumable_training_state(args, frozen.number)
        print(
            f"[OPTUNA-ISO] resuming RUNNING trial={frozen.number} "
            f"from {'saved training_state' if has_state else 'scratch/same trial directory'}"
        )
        completed = _run_worker_for_trial(
            study=study,
            args=args,
            trial_ref=frozen.number,
            trial_number=int(frozen.number),
            params=params,
            trial_id=_trial_id_from_trial_or_frozen(frozen),
        )
        made_progress = made_progress or completed
        if not completed:
            raise SystemExit(143)
    return made_progress


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-case staged Optuna for 1-DMRS UPAIR ablations with one TensorFlow subprocess per trial."
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Safety cap on trial attempts in this controller invocation. Defaults to --n-trials.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    validate_common_args(args)
    args.trial_process_isolation = True
    if args.max_attempts is None:
        args.max_attempts = int(args.n_trials)

    storage = optuna.storages.RDBStorage(
        url=args.storage,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=make_sampler(args),
        pruner=make_pruner(args),
    )

    previous_handlers = _install_signal_handlers()
    try:
        if int(args.source_top_k) > 0:
            enqueue_stage_candidates(study, args)

        _resume_running_trials(study, args)

        completed_before = completed_trial_count(study)
        pruned_before = pruned_trial_count(study)
        finished_before = finished_trial_count(study)
        failed_before = failed_trial_count(study)
        remaining_trials = max(0, int(args.target_total_trials) - finished_before)
        attempts_to_run = min(int(args.max_attempts), int(args.n_trials), remaining_trials)
        print(
            "[OPTUNA-ISO] resume_state "
            f"stage={args.stage} "
            f"completed_trials={completed_before} "
            f"pruned_trials={pruned_before} "
            f"finished_trials={finished_before} "
            f"failed_trials={failed_before} "
            f"nonfailed_trials={nonfailed_trial_count(study)} "
            f"n_trials_request={int(args.n_trials)} "
            f"max_attempts={int(args.max_attempts)} "
            f"target_total_trials={int(args.target_total_trials)} "
            f"attempts_to_run={attempts_to_run} "
            f"steps={int(args.steps)} "
            "trial_process_isolation=1"
        )

        attempts = 0
        while attempts < attempts_to_run and finished_trial_count(study) < int(args.target_total_trials):
            if _STOP_REQUESTED:
                raise SystemExit(143)
            trial = study.ask()
            trial_id = _trial_id_from_trial_or_frozen(trial)
            params = suggest_trial_params(args, trial)
            params = normalize_optuna_params(args, params)
            print(f"[OPTUNA-ISO] starting stage={args.stage} trial={trial.number} trial_id={trial_id} params={params}")
            attempts += 1
            completed = _run_worker_for_trial(
                study=study,
                args=args,
                trial_ref=trial,
                trial_number=int(trial.number),
                params=params,
                trial_id=int(trial_id),
            )
            if not completed:
                raise SystemExit(143)

        print("[OPTUNA-ISO] study:", args.study_name)
        print("[OPTUNA-ISO] stage:", args.stage)
        print("[OPTUNA-ISO] trials:", len(study.trials))
        print("[OPTUNA-ISO] completed_trials:", completed_trial_count(study))
        print("[OPTUNA-ISO] pruned_trials:", pruned_trial_count(study))
        print("[OPTUNA-ISO] finished_trials:", finished_trial_count(study))
        print("[OPTUNA-ISO] failed_trials:", failed_trial_count(study))
        if completed_trial_count(study) > 0:
            print("[OPTUNA-ISO] best_value:", study.best_value)
            print("[OPTUNA-ISO] best_params:", normalize_optuna_params(args, study.best_params))
        export_best_params(study, args)
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    main()
