from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import load_config, set_cfg  # noqa: E402

VARIANTS: dict[str, dict[str, Any]] = {
    "main_d256_b4_r2": {"model.d_model": 256, "model.num_blocks": 4, "model.mlp_ratio": 2.0},
    "shallow_d256_b2_r2": {"model.d_model": 256, "model.num_blocks": 2, "model.mlp_ratio": 2.0},
    "deep_d256_b6_r2": {"model.d_model": 256, "model.num_blocks": 6, "model.mlp_ratio": 2.0},
    "narrow_d192_b4_r2": {"model.d_model": 192, "model.num_blocks": 4, "model.mlp_ratio": 2.0},
    "wide_d320_b4_r2": {"model.d_model": 320, "model.num_blocks": 4, "model.mlp_ratio": 2.0},
    "wide_deep_d320_b6_r2": {"model.d_model": 320, "model.num_blocks": 6, "model.mlp_ratio": 2.0},
    "mlpwide_d256_b4_r4": {"model.d_model": 256, "model.num_blocks": 4, "model.mlp_ratio": 4.0},
}

SUGGESTED_PARAM_NAMES = {
    "learning_rate_schedule",
    "learning_rate",
    "learning_rate_decay_fraction",
    "learning_rate_final_fraction",
    "learning_rate_polynomial_power",
    "weight_decay",
    "nmse_loss_weight",
    "grad_clip_norm",
    "dropout",
    "residual_scale",
}
ESSENTIAL_PARAM_NAMES = {
    "learning_rate_schedule",
    "learning_rate",
    "weight_decay",
    "nmse_loss_weight",
    "grad_clip_norm",
    "dropout",
    "residual_scale",
}
STAGE_DEFAULTS: dict[str, dict[str, int]] = {
    # Smart bounded schedule for the PRB8/d256 package:
    # Stage A explores broadly but cheaply; Stage B re-runs the best candidates longer.
    "A": {"steps": 4000, "target_total_trials": 20, "source_top_k": 0},
    "B": {"steps": 12000, "target_total_trials": 6, "source_top_k": 6},
    "C": {"steps": 40000, "target_total_trials": 3, "source_top_k": 3},
}


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return values


def completed_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)


def pruned_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED)


def finished_trial_count(study: optuna.Study) -> int:
    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    return sum(1 for trial in study.trials if trial.state in finished_states)


def failed_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL)


def nonfailed_trial_count(study: optuna.Study) -> int:
    useful_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    }
    return sum(1 for trial in study.trials if trial.state in useful_states)


def normalize_optuna_params(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    result = dict(params)
    result["batch_size_train"] = int(args.train_batch_size)
    result["batch_size_eval"] = int(args.validation_batch_size)
    result.setdefault("learning_rate_schedule", "cosine_decay")
    result.setdefault("learning_rate_decay_fraction", 1.0)
    result.setdefault("learning_rate_final_fraction", 0.05)
    result.setdefault("learning_rate_polynomial_power", 1.0)
    result.setdefault("grad_clip_norm", 1.0)
    return result


def candidate_params_for_enqueue(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_optuna_params(args, params)
    schedule = str(normalized.get("learning_rate_schedule", "cosine_decay"))
    names = {
        "learning_rate_schedule",
        "learning_rate",
        "weight_decay",
        "nmse_loss_weight",
        "grad_clip_norm",
        "dropout",
        "residual_scale",
    }
    if schedule in {"cosine_decay", "polynomial_decay"}:
        names.update({"learning_rate_decay_fraction", "learning_rate_final_fraction"})
    if schedule == "polynomial_decay":
        names.add("learning_rate_polynomial_power")
    return {name: normalized[name] for name in names if name in normalized}


def suggest_trial_params(args: argparse.Namespace, trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "batch_size_train": int(args.train_batch_size),
        "batch_size_eval": int(args.validation_batch_size),
    }
    if getattr(args, "smoke", False):
        params.update(
            {
                "learning_rate_schedule": "constant",
                "learning_rate": 3e-4,
                "learning_rate_decay_fraction": 1.0,
                "learning_rate_final_fraction": 0.05,
                "learning_rate_polynomial_power": 1.0,
                "weight_decay": 1e-5,
                "nmse_loss_weight": 0.1,
                "grad_clip_norm": 1.0,
                "dropout": 0.05,
                "residual_scale": 0.35,
            }
        )
        return params

    schedule = trial.suggest_categorical(
        "learning_rate_schedule",
        ["constant", "cosine_decay", "polynomial_decay"],
    )
    params["learning_rate_schedule"] = schedule
    params["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 1.2e-3, log=True)

    if schedule in {"cosine_decay", "polynomial_decay"}:
        params["learning_rate_decay_fraction"] = trial.suggest_categorical(
            "learning_rate_decay_fraction",
            [0.75, 1.0, 1.25],
        )
        params["learning_rate_final_fraction"] = trial.suggest_float(
            "learning_rate_final_fraction",
            0.01,
            0.15,
            log=True,
        )
    else:
        params["learning_rate_decay_fraction"] = 1.0
        params["learning_rate_final_fraction"] = 0.05

    if schedule == "polynomial_decay":
        params["learning_rate_polynomial_power"] = trial.suggest_float(
            "learning_rate_polynomial_power",
            0.7,
            2.0,
        )
    else:
        params["learning_rate_polynomial_power"] = 1.0

    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 2e-4, log=True)
    params["nmse_loss_weight"] = trial.suggest_float("nmse_loss_weight", 0.03, 0.5, log=True)
    params["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0])
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.12)
    params["residual_scale"] = trial.suggest_float("residual_scale", 0.15, 0.75)
    return params


def force_clean_optuna_common_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    set_cfg(cfg, "impairments.enabled", False)
    set_cfg(cfg, "legacy_phase_impairments.enabled", False)
    set_cfg(cfg, "rf_impairments.enabled", False)
    set_cfg(cfg, "rf_impairments.train_mixture.enabled", False)
    set_cfg(cfg, "rf_impairments.cfo.enabled", False)
    set_cfg(cfg, "rf_impairments.phase_noise.enabled", False)

    set_cfg(cfg, "system.batch_size_train", int(args.train_batch_size))
    set_cfg(cfg, "system.batch_size_eval", int(args.validation_batch_size))

    set_cfg(cfg, "multiuser.enabled", True)
    set_cfg(cfg, "multiuser.max_num_users", 4)
    set_cfg(cfg, "multiuser.train_user_count_sampler", "weighted")
    set_cfg(cfg, "multiuser.train_user_count_weights", [float(x) for x in args.train_user_count_weights])
    set_cfg(cfg, "system.ebno_db_train_min", float(args.train_ebno_min))
    set_cfg(cfg, "system.ebno_db_train_max", float(args.train_ebno_max))

    set_cfg(cfg, "training.val_sampling_mode", "sampled")
    set_cfg(cfg, "training.val_snr_sampling_mode", "sampled_grid")
    set_cfg(cfg, "training.val_user_counts", [int(x) for x in args.val_user_counts])
    set_cfg(cfg, "training.val_user_count_weights", [float(x) for x in args.val_user_count_weights])
    set_cfg(cfg, "training.val_microbatch_size", int(args.validation_microbatch_size))
    set_cfg(cfg, "training.val_memory_cleanup_every_microbatch", bool(args.val_memory_cleanup_every_batch))
    set_cfg(cfg, "training.val_memory_cleanup_every_batch", bool(args.val_memory_cleanup_every_batch))
    set_cfg(cfg, "training.memory_cleanup_after_validation", True)
    set_cfg(cfg, "training.memory_cleanup_every_steps", int(args.memory_cleanup_every_steps))

    # Keep the final/evaluation streaming settings intact and explicitly safe in resolved configs.
    set_cfg(cfg, "evaluation.receiver_microbatch_size", 16)
    set_cfg(cfg, "evaluation.stream_eval_microbatches", True)
    set_cfg(cfg, "evaluation.compiled_receiver_error_counts", True)
    set_cfg(cfg, "evaluation.memory_cleanup_every_microbatch", True)
    set_cfg(cfg, "evaluation.memory_cleanup_every_batches", 1)


def apply_params_cfg(cfg: dict[str, Any], args: argparse.Namespace, params: dict[str, Any], trial_number: int) -> None:
    params = normalize_optuna_params(args, params)
    for path, value in VARIANTS[args.variant].items():
        set_cfg(cfg, path, value)

    set_cfg(cfg, "multiuser.dmrs.length", 1)
    set_cfg(cfg, "multiuser.dmrs.additional_position", 0)

    set_cfg(cfg, "system.seed", int(args.seed))
    set_cfg(cfg, "system.training_seed", int(args.seed))
    set_cfg(cfg, "system.evaluation_seed", int(args.seed) + 1000)
    force_clean_optuna_common_cfg(cfg, args)

    set_cfg(cfg, "training.steps", int(args.steps))
    set_cfg(cfg, "training.eval_every", min(int(args.eval_every), int(args.steps)))
    set_cfg(cfg, "training.checkpoint_every", min(int(args.checkpoint_every), int(args.steps)))
    set_cfg(cfg, "training.log_every", min(int(args.log_every), int(args.steps)))
    set_cfg(cfg, "training.val_steps", int(args.val_steps))
    set_cfg(cfg, "training.val_ebno_db", [float(x) for x in args.val_ebno_db])
    set_cfg(cfg, "training.resume", True)

    set_cfg(cfg, "training.learning_rate_schedule", str(params["learning_rate_schedule"]))
    set_cfg(cfg, "training.learning_rate", float(params["learning_rate"]))
    set_cfg(cfg, "training.learning_rate_decay_steps", max(1, int(int(args.steps) * float(params["learning_rate_decay_fraction"]))))
    set_cfg(cfg, "training.learning_rate_final_fraction", float(params["learning_rate_final_fraction"]))
    set_cfg(cfg, "training.learning_rate_polynomial_power", float(params["learning_rate_polynomial_power"]))
    set_cfg(cfg, "training.weight_decay", float(params["weight_decay"]))
    set_cfg(cfg, "training.nmse_loss_weight", float(params["nmse_loss_weight"]))
    set_cfg(cfg, "training.grad_clip_norm", float(params["grad_clip_norm"]))
    set_cfg(cfg, "model.dropout", float(params["dropout"]))
    set_cfg(cfg, "model.residual_scale", float(params["residual_scale"]))

    set_cfg(cfg, "experiment.output_root", f"optuna/runs_1dmrs/{args.study_name}")
    set_cfg(cfg, "experiment.name", f"{args.variant}_trial_{int(trial_number):04d}")


def make_trial_config(args: argparse.Namespace, params: dict[str, Any], trial_number: int) -> dict[str, Any]:
    cfg = load_config(args.config)
    apply_params_cfg(cfg, args, params, trial_number)
    return cfg


def score_validation_row(row: dict[str, Any], objective_metric: str) -> float:
    val_nmse_prop = float(row["val_nmse_prop"])
    val_nmse_ls = float(row.get("val_nmse_ls", val_nmse_prop))
    ratio = val_nmse_prop / max(val_nmse_ls, 1e-12)
    if objective_metric == "prop_nmse":
        return math.log10(max(val_nmse_prop, 1e-12))
    if objective_metric == "prop_nmse_ratio":
        return math.log10(max(ratio, 1e-12))
    if objective_metric == "hybrid_nmse_ratio":
        return math.log10(max(val_nmse_prop, 1e-12)) + 0.20 * math.log10(max(ratio, 1e-12))
    raise ValueError(f"Unknown objective metric {objective_metric!r}")


def validation_history_score(
    history_path: Path,
    objective_metric: str,
    aggregation: str = "recent_mean",
    recent_k: int = 2,
    min_step: int = 0,
) -> float:
    with open(history_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = [
        dict(row)
        for row in payload.get("history", [])
        if isinstance(row, dict) and "val_nmse_prop" in row and int(row.get("step", 0)) > int(min_step)
    ]
    if not rows:
        raise RuntimeError(f"No validation metrics found in {history_path}")
    values = [score_validation_row(row, objective_metric) for row in rows]
    if aggregation == "best":
        return float(min(values))
    if aggregation == "last":
        return float(values[-1])
    if aggregation == "recent_mean":
        k = max(1, min(int(recent_k), len(values)))
        return float(sum(values[-k:]) / k)
    raise ValueError(f"Unknown objective aggregation {aggregation!r}")


def trial_output_root(args: argparse.Namespace, trial_number: int) -> Path:
    return PROJECT_ROOT / "optuna" / "runs_1dmrs" / args.study_name / f"{args.variant}_trial_{int(trial_number):04d}"


def read_train_state(args: argparse.Namespace, trial_number: int) -> dict[str, Any] | None:
    state_path = trial_output_root(args, trial_number) / "metrics" / "train_state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload) if isinstance(payload, dict) else None
    except Exception:
        return None


def history_path_for_trial(args: argparse.Namespace, trial_number: int) -> Path:
    return trial_output_root(args, trial_number) / "metrics" / "history.json"


def has_resumable_training_state(args: argparse.Namespace, trial_number: int) -> bool:
    payload = read_train_state(args, trial_number)
    if payload is None:
        return False
    if bool(payload.get("training_complete", False)) or bool(payload.get("optuna_pruned", False)):
        return False
    return True


def export_best_params(study: optuna.Study, args: argparse.Namespace) -> None:
    if completed_trial_count(study) <= 0:
        print("[OPTUNA] no completed trials; skipping best-params export.")
        return
    best_path = PROJECT_ROOT / "optuna" / f"{args.study_name}_best_params.json"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_best = normalize_optuna_params(args, study.best_params)
    best_params = {
        key: value
        for key, value in normalized_best.items()
        if key not in {"batch_size_train", "batch_size_eval"}
    }

    payload = {
        "study_name": args.study_name,
        "stage": args.stage,
        "completed_trials": completed_trial_count(study),
        "pruned_trials": pruned_trial_count(study),
        "finished_trials": finished_trial_count(study),
        "failed_trials": failed_trial_count(study),
        "target_total_trials": int(args.target_total_trials),
        "best_value": float(study.best_value),
        "best_params": best_params,
        "objective_metric": args.objective_metric,
        "objective_aggregation": args.objective_aggregation,
        "objective_recent_k": int(args.objective_recent_k),
        "objective_min_step_exclusive": int(args.objective_min_step),
        "steps": int(args.steps),
        "val_steps": int(args.val_steps),
        "val_ebno_db": [float(x) for x in args.val_ebno_db],
        "val_user_counts": [int(x) for x in args.val_user_counts],
        "val_user_count_weights": [float(x) for x in args.val_user_count_weights],
        "train_user_count_weights": [float(x) for x in args.train_user_count_weights],
        "fixed_constraints": {
            "clean_case": True,
            "dmrs_length": 1,
            "train_batch_size": int(args.train_batch_size),
            "optuna_validation_batch_size": int(args.validation_batch_size),
            "optuna_validation_microbatch_size": int(args.validation_microbatch_size),
            "evaluation_receiver_microbatch_size": 16,
            "evaluation_stream_eval_microbatches": True,
            "trial_process_isolation": bool(getattr(args, "trial_process_isolation", False)),
            "resource_exhausted_counts_as_finished": False,
        },
    }
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[OPTUNA] exported best params: {best_path}")


def params_from_frozen_trial(args: argparse.Namespace, frozen: optuna.trial.FrozenTrial) -> dict[str, Any]:
    params = dict(frozen.params)
    if not params:
        fixed = getattr(frozen, "system_attrs", {}).get("fixed_params", {})
        if isinstance(fixed, dict):
            params = dict(fixed)
    return candidate_params_for_enqueue(args, params)


def source_storage_url(args: argparse.Namespace) -> str:
    if args.source_storage:
        return str(args.source_storage)
    if not args.source_study_name:
        raise ValueError("A source study name/storage is required for staged candidate promotion.")
    return f"sqlite:///{(PROJECT_ROOT / 'optuna' / (args.source_study_name + '.db')).resolve()}"


def load_source_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if int(args.source_top_k) <= 0:
        return []
    if not args.source_study_name:
        raise ValueError(f"Stage {args.stage} requires --source-study-name or OPTUNA_SOURCE_STUDY_NAME.")
    source = optuna.load_study(study_name=args.source_study_name, storage=source_storage_url(args))
    completed = [trial for trial in source.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    completed.sort(key=lambda t: float(t.value))
    if len(completed) < int(args.source_top_k):
        raise RuntimeError(
            f"Source study {args.source_study_name!r} has only {len(completed)} completed trials; "
            f"need {int(args.source_top_k)} for stage {args.stage}."
        )
    candidates = [params_from_frozen_trial(args, frozen) for frozen in completed[: int(args.source_top_k)]]
    print(f"[OPTUNA] loaded {len(candidates)} candidates from source study={args.source_study_name}")
    return candidates


def param_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def candidate_already_present(study: optuna.Study, candidate: dict[str, Any]) -> bool:
    for trial in study.get_trials(deepcopy=False):
        params = dict(trial.params)
        if not params:
            fixed = getattr(trial, "system_attrs", {}).get("fixed_params", {})
            if isinstance(fixed, dict):
                params = dict(fixed)
        if not params:
            continue
        if all(name in params and param_equal(params[name], value) for name, value in candidate.items()):
            return True
    return False


def enqueue_stage_candidates(study: optuna.Study, args: argparse.Namespace) -> None:
    candidates = load_source_candidates(args)
    if not candidates:
        return
    for rank, candidate in enumerate(candidates, start=1):
        if candidate_already_present(study, candidate):
            continue
        user_attrs = {"source_study_name": args.source_study_name, "source_rank": rank, "stage": args.stage}
        try:
            study.enqueue_trial(candidate, user_attrs=user_attrs, skip_if_exists=True)
        except TypeError:
            study.enqueue_trial(candidate, user_attrs=user_attrs)
        print(f"[OPTUNA] enqueued stage={args.stage} source_rank={rank} params={candidate}")


def make_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.disable_pruning or args.pruner == "none" or args.stage in {"B", "C"}:
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=int(args.prune_warmup_steps),
            interval_steps=int(args.prune_interval_steps),
            n_min_trials=int(args.pruner_min_trials),
        )
    if args.pruner == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=max(1, int(args.prune_interval_steps)),
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    if args.pruner == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=float(args.pruner_percentile),
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=int(args.prune_warmup_steps),
            interval_steps=int(args.prune_interval_steps),
            n_min_trials=int(args.pruner_min_trials),
        )
    raise ValueError(f"Unknown pruner {args.pruner!r}")


def make_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(
        seed=int(args.seed),
        multivariate=True,
        group=True,
        n_startup_trials=int(args.tpe_startup_trials),
        constant_liar=bool(args.constant_liar),
    )


def apply_stage_defaults(args: argparse.Namespace) -> None:
    args.stage = str(args.stage).upper()
    if args.stage not in STAGE_DEFAULTS:
        raise ValueError(f"Unknown stage {args.stage!r}; expected A, B, or C.")
    defaults = STAGE_DEFAULTS[args.stage]
    if args.steps is None:
        args.steps = int(defaults["steps"])
    if args.target_total_trials is None:
        args.target_total_trials = int(defaults["target_total_trials"])
    if args.n_trials is None:
        args.n_trials = int(args.target_total_trials)
    if args.source_top_k is None:
        args.source_top_k = int(defaults["source_top_k"])


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--stage", default="A", choices=["A", "B", "C", "a", "b", "c"])
    parser.add_argument("--source-study-name", default=None)
    parser.add_argument("--source-storage", default=None)
    parser.add_argument("--source-top-k", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--target-total-trials", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-steps", type=int, default=96)
    parser.add_argument("--val-ebno-db", type=parse_float_list, default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    parser.add_argument("--val-user-counts", type=parse_int_list, default=[1, 2, 3, 4])
    parser.add_argument("--val-user-count-weights", type=parse_float_list, default=[1.0, 3.0, 6.0, 10.0])
    parser.add_argument("--train-user-count-weights", type=parse_float_list, default=[1.0, 3.0, 6.0, 10.0])
    parser.add_argument("--train-ebno-min", type=float, default=-6.0)
    parser.add_argument("--train-ebno-max", type=float, default=5.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--validation-microbatch-size", type=int, default=16)
    parser.add_argument(
        "--objective-metric",
        choices=["prop_nmse", "prop_nmse_ratio", "hybrid_nmse_ratio"],
        default="hybrid_nmse_ratio",
    )
    parser.add_argument("--objective-aggregation", choices=["recent_mean", "last", "best"], default="recent_mean")
    parser.add_argument("--objective-recent-k", type=int, default=2)
    parser.add_argument("--objective-min-step", type=int, default=1000, help="Validation rows with step <= this value are excluded from the final Optuna objective.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tpe-startup-trials", type=int, default=8)
    parser.add_argument("--constant-liar", action="store_true")
    parser.add_argument("--pruner", choices=["percentile", "median", "successive_halving", "none"], default="percentile")
    parser.add_argument("--disable-pruning", action="store_true")
    parser.add_argument("--pruner-percentile", type=float, default=25.0)
    parser.add_argument("--pruner-startup-trials", type=int, default=6)
    parser.add_argument("--pruner-min-trials", type=int, default=4)
    parser.add_argument("--prune-warmup-steps", type=int, default=1000)
    parser.add_argument("--prune-interval-steps", type=int, default=1000)
    parser.add_argument("--memory-cleanup-every-steps", type=int, default=100)
    parser.add_argument("--val-memory-cleanup-every-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")


def validate_common_args(args: argparse.Namespace) -> None:
    apply_stage_defaults(args)
    if bool(getattr(args, "smoke", False)) and int(getattr(args, "objective_min_step", 0)) > 0:
        # Smoke probes often run only 1--2 steps; do not exclude all validation rows.
        args.objective_min_step = 0
    if int(args.train_batch_size) <= 0 or int(args.validation_batch_size) <= 0:
        raise ValueError("Batch sizes must be positive.")
    if int(args.train_batch_size) > 32 or int(args.validation_batch_size) > 32:
        raise ValueError("This Narval 40GB package caps Optuna training/validation batch sizes at 32 for safety.")
    if int(args.validation_microbatch_size) <= 0:
        raise ValueError("--validation-microbatch-size must be positive.")
    if int(args.validation_microbatch_size) > int(args.validation_batch_size):
        raise ValueError("--validation-microbatch-size must be <= --validation-batch-size.")
    if len(args.val_user_counts) != len(args.val_user_count_weights):
        raise ValueError("--val-user-counts and --val-user-count-weights must have the same length.")
    if len(args.train_user_count_weights) != 4:
        raise ValueError("--train-user-count-weights must contain four weights for users 1,2,3,4.")
