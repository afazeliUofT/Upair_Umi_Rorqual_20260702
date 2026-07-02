from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402
from upair5g.training import train_model  # noqa: E402


VARIANTS: dict[str, dict[str, Any]] = {
    "main_d256_b4_r2": {
        "label": "d=256, L=4, r=2",
        "overrides": {
            "model.d_model": 256,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
    "shallow_d256_b2_r2": {
        "label": "d=256, L=2, r=2",
        "overrides": {
            "model.d_model": 256,
            "model.num_blocks": 2,
            "model.mlp_ratio": 2.0,
        },
    },
    "deep_d256_b6_r2": {
        "label": "d=256, L=6, r=2",
        "overrides": {
            "model.d_model": 256,
            "model.num_blocks": 6,
            "model.mlp_ratio": 2.0,
        },
    },
    "narrow_d192_b4_r2": {
        "label": "d=192, L=4, r=2",
        "overrides": {
            "model.d_model": 192,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
    "wide_d320_b4_r2": {
        "label": "d=320, L=4, r=2",
        "overrides": {
            "model.d_model": 320,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
    "wide_deep_d320_b6_r2": {
        "label": "d=320, L=6, r=2",
        "overrides": {
            "model.d_model": 320,
            "model.num_blocks": 6,
            "model.mlp_ratio": 2.0,
        },
    },
    "mlpwide_d256_b4_r4": {
        "label": "d=256, L=4, r=4",
        "overrides": {
            "model.d_model": 256,
            "model.num_blocks": 4,
            "model.mlp_ratio": 4.0,
        },
    },
}


DMRS_CASES: dict[str, dict[str, Any]] = {
    "1dmrs": {
        "label": "1-DMRS",
        "overrides": {
            "multiuser.dmrs.length": 1,
            "multiuser.dmrs.additional_position": 0,
        },
    },
    "2dmrs": {
        "label": "2-DMRS",
        "overrides": {
            "multiuser.dmrs.length": 1,
            "multiuser.dmrs.additional_position": 1,
        },
    },
}


OPTUNA_BEST_1DMRS: dict[str, dict[str, Any]] = {
    "main_d256_b4_r2": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0005950487024755413,
        "training.weight_decay": 7.169815362007231e-05,
        "training.nmse_loss_weight": 0.2313047274659292,
        "model.dropout": 0.0010714912918497743,
        "model.residual_scale": 0.39417978357077454,
    },
    "shallow_d256_b2_r2": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0005950487024755413,
        "training.weight_decay": 7.169815362007231e-05,
        "training.nmse_loss_weight": 0.2313047274659292,
        "model.dropout": 0.0010714912918497743,
        "model.residual_scale": 0.39417978357077454,
    },
    "deep_d256_b6_r2": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0005950487024755413,
        "training.weight_decay": 7.169815362007231e-05,
        "training.nmse_loss_weight": 0.2313047274659292,
        "model.dropout": 0.0010714912918497743,
        "model.residual_scale": 0.39417978357077454,
    },
    "narrow_d192_b4_r2": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0006975764648386961,
        "training.weight_decay": 5.876882839088582e-05,
        "training.nmse_loss_weight": 0.1074731692161375,
        "model.dropout": 0.06909454442449153,
        "model.residual_scale": 0.6447495210807277,
    },
    "wide_d320_b4_r2": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0005950487024755413,
        "training.weight_decay": 7.169815362007231e-05,
        "training.nmse_loss_weight": 0.2313047274659292,
        "model.dropout": 0.0010714912918497743,
        "model.residual_scale": 0.39417978357077454,
    },
    "wide_deep_d320_b6_r2": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0005950487024755413,
        "training.weight_decay": 7.169815362007231e-05,
        "training.nmse_loss_weight": 0.2313047274659292,
        "model.dropout": 0.0010714912918497743,
        "model.residual_scale": 0.39417978357077454,
    },
    "mlpwide_d256_b4_r4": {
        "system.batch_size_train": 32,
        "system.batch_size_eval": 32,
        "training.learning_rate": 0.0005950487024755413,
        "training.weight_decay": 7.169815362007231e-05,
        "training.nmse_loss_weight": 0.2313047274659292,
        "model.dropout": 0.0010714912918497743,
        "model.residual_scale": 0.39417978357077454,
    },
}


def _optuna_params_to_overrides(params: dict[str, Any]) -> dict[str, Any]:
    direct_map = {
        "batch_size_train": "system.batch_size_train",
        "batch_size_train_safe": "system.batch_size_train",
        "learning_rate": "training.learning_rate",
        "learning_rate_schedule": "training.learning_rate_schedule",
        "learning_rate_final_fraction": "training.learning_rate_final_fraction",
        "learning_rate_polynomial_power": "training.learning_rate_polynomial_power",
        "learning_rate_decay_rate": "training.learning_rate_decay_rate",
        "learning_rate_restart_t_mul": "training.learning_rate_restart_t_mul",
        "learning_rate_restart_m_mul": "training.learning_rate_restart_m_mul",
        "weight_decay": "training.weight_decay",
        "nmse_loss_weight": "training.nmse_loss_weight",
        "grad_clip_norm": "training.grad_clip_norm",
        "dropout": "model.dropout",
        "residual_scale": "model.residual_scale",
    }
    overrides: dict[str, Any] = {
        cfg_path: params[name]
        for name, cfg_path in direct_map.items()
        if name in params
    }
    if "learning_rate_decay_fraction" in params:
        overrides["training.learning_rate_decay_fraction"] = float(params["learning_rate_decay_fraction"])
    if "learning_rate_restart_fraction" in params:
        overrides["training.learning_rate_restart_fraction"] = float(params["learning_rate_restart_fraction"])
    return overrides


def _load_optuna_best_overrides(
    variant_name: str,
    storage_dir: str | Path | None,
    study_prefix: str,
) -> dict[str, Any] | None:
    if storage_dir is None or str(storage_dir).strip().lower() in {"", "none", "null"}:
        return None
    storage_dir = Path(storage_dir)
    study_name = f"{study_prefix}_{variant_name}"

    # Prefer the JSON artifact exported by scripts/run_optuna_1dmrs_structure.py.
    # This avoids importing Optuna during the final train/eval pass and avoids
    # SQLite locking surprises on shared filesystems.
    json_path = storage_dir / f"{study_name}_best_params.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            params = dict(payload.get("best_params", {}))
            if params:
                print(f"[COMPREHENSIVE] loaded Optuna best JSON for {variant_name} from {json_path}")
                return _optuna_params_to_overrides(params)
        except Exception as exc:
            print(f"[COMPREHENSIVE] could not load Optuna best JSON from {json_path}: {exc!r}")

    db_path = storage_dir / f"{study_name}.db"
    if not db_path.exists():
        return None
    try:
        import optuna

        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path.resolve()}",
        )
        completed = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            print(f"[COMPREHENSIVE] Optuna DB has no completed trials yet: {db_path}")
            return None
    except Exception as exc:
        print(f"[COMPREHENSIVE] could not load Optuna best from {db_path}: {exc!r}")
        return None
    print(f"[COMPREHENSIVE] loaded Optuna best for {variant_name} from {db_path}")
    return _optuna_params_to_overrides(study.best_params)

def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    for path, value in overrides.items():
        set_cfg(result, path, value)
    return result


def _rx_tag(cfg: dict[str, Any]) -> str:
    return f"rx{int(get_cfg(cfg, 'channel.num_rx_ant', 0))}"


def _seed_tag(seed: int) -> str:
    return f"seed{int(seed)}"


def _eval_seed_from_train_seed(cfg: dict[str, Any], train_seed: int) -> int:
    return int(train_seed) + int(get_cfg(cfg, "system.eval_seed_offset", 1000))


def _parse_seed_values(base_cfg: dict[str, Any], seed_arg: str | None) -> list[int]:
    if seed_arg:
        values = [int(x.strip()) for x in seed_arg.split(",") if x.strip()]
    else:
        configured = get_cfg(base_cfg, "system.seeds", None)
        if configured is None:
            configured = [get_cfg(base_cfg, "system.seed", 7)]
        if isinstance(configured, str):
            values = [int(x.strip()) for x in configured.split(",") if x.strip()]
        elif isinstance(configured, (int, float)):
            values = [int(configured)]
        else:
            values = [int(x) for x in configured]
    if not values:
        raise ValueError("At least one seed must be configured.")
    return values


def _case_cfg(base_cfg: dict[str, Any], dmrs_case: str) -> dict[str, Any]:
    if dmrs_case not in DMRS_CASES:
        raise KeyError(f"Unknown DMRS case {dmrs_case}. Available: {sorted(DMRS_CASES)}")
    return _apply_overrides(base_cfg, DMRS_CASES[dmrs_case]["overrides"])


def _variant_cfg(base_cfg: dict[str, Any], variant_name: str, dmrs_case: str, seed: int) -> dict[str, Any]:
    if variant_name not in VARIANTS:
        raise KeyError(f"Unknown variant {variant_name}. Available: {sorted(VARIANTS)}")
    cfg = _case_cfg(base_cfg, dmrs_case)
    cfg = _apply_overrides(cfg, VARIANTS[variant_name]["overrides"])
    set_cfg(cfg, "system.seed", int(seed))
    set_cfg(cfg, "system.training_seed", int(seed))
    set_cfg(cfg, "system.evaluation_seed", _eval_seed_from_train_seed(cfg, int(seed)))
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/runs_{_rx_tag(cfg)}/{_seed_tag(seed)}/{dmrs_case}")
    set_cfg(cfg, "experiment.name", variant_name)
    return cfg


def _apply_optuna_best_1dmrs(
    cfg: dict[str, Any],
    variant_name: str,
    dmrs_case: str,
    storage_dir: str | Path | None = None,
    study_prefix: str = "clean_b32_prb8_d256_u34610_1dmrs_stageC",
    require_external: bool = False,
) -> None:
    if dmrs_case != "1dmrs":
        raise ValueError("--use-optuna-best-1dmrs is only valid with dmrs_case=1dmrs.")
    overrides = _load_optuna_best_overrides(variant_name, storage_dir, study_prefix)
    if overrides is None:
        if require_external:
            raise FileNotFoundError(
                f"No completed Optuna best study found for {variant_name} in {storage_dir} "
                f"with prefix {study_prefix!r}. Run Optuna first or unset --require-optuna-best."
            )
        if variant_name not in OPTUNA_BEST_1DMRS:
            raise KeyError(f"No built-in Optuna best parameters recorded for variant {variant_name}.")
        overrides = dict(OPTUNA_BEST_1DMRS[variant_name])
        if storage_dir is not None:
            print(
                f"[COMPREHENSIVE] no Optuna DB best found for {variant_name} in {storage_dir}; "
                "falling back to built-in Optuna defaults. Set --require-optuna-best to make this fallback an error."
            )
        else:
            print(f"[COMPREHENSIVE] using built-in Optuna best defaults for {variant_name}")
    steps = int(get_cfg(cfg, "training.steps", 10000))
    if "training.learning_rate_decay_fraction" in overrides:
        overrides["training.learning_rate_decay_steps"] = max(
            1,
            int(steps * float(overrides.pop("training.learning_rate_decay_fraction"))),
        )
    if "training.learning_rate_restart_fraction" in overrides:
        overrides["training.learning_rate_restart_first_decay_steps"] = max(
            1,
            int(steps * float(overrides.pop("training.learning_rate_restart_fraction"))),
        )
    # Safety guard: Optuna-best application must never reintroduce the old unsafe large-batch settings.
    overrides["system.batch_size_train"] = 32
    overrides["system.batch_size_eval"] = 32
    overrides["training.val_microbatch_size"] = 16
    for path, value in overrides.items():
        set_cfg(cfg, path, value)
    # Preserve evaluation memory controls from the YAML/config.
    # Final OOM-safe workflow keeps receiver_microbatch_size=16, disables explicit NMSE,
    # uses compiled receiver error counts, and runs evaluation one Eb/N0 per process.
    eval_logical_batch = int(get_cfg(cfg, "evaluation.logical_batch_size", get_cfg(cfg, "system.batch_size_eval", 32)))
    eval_microbatch = int(get_cfg(cfg, "evaluation.receiver_microbatch_size", min(16, eval_logical_batch)))
    set_cfg(cfg, "evaluation.receiver_microbatch_size", max(1, min(eval_microbatch, eval_logical_batch)))
    set_cfg(cfg, "evaluation.stream_eval_microbatches", bool(get_cfg(cfg, "evaluation.stream_eval_microbatches", True)))
    set_cfg(cfg, "evaluation.compiled_receiver_error_counts", bool(get_cfg(cfg, "evaluation.compiled_receiver_error_counts", True)))
    set_cfg(cfg, "evaluation.memory_cleanup_every_microbatch", bool(get_cfg(cfg, "evaluation.memory_cleanup_every_microbatch", True)))
    set_cfg(cfg, "evaluation.memory_cleanup_every_batches", int(get_cfg(cfg, "evaluation.memory_cleanup_every_batches", 1)))


def _eval_cfg(train_cfg: dict[str, Any], variant_name: str, dmrs_case: str, num_users: int) -> dict[str, Any]:
    cfg = copy.deepcopy(train_cfg)
    training_seed = int(get_cfg(cfg, "system.training_seed", get_cfg(cfg, "system.seed", 0)))
    evaluation_seed = int(get_cfg(cfg, "system.evaluation_seed", _eval_seed_from_train_seed(cfg, training_seed)))
    set_cfg(cfg, "system.training_seed", training_seed)
    set_cfg(cfg, "system.evaluation_seed", evaluation_seed)
    set_cfg(cfg, "system.seed", evaluation_seed)
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/eval_runs_{_rx_tag(cfg)}/{_seed_tag(training_seed)}/{dmrs_case}")
    set_cfg(cfg, "experiment.name", f"{variant_name}_u{num_users}")
    set_cfg(cfg, "multiuser.fixed_num_users", int(num_users))
    # Final Monte-Carlo eval uses a logical batch, but keep the actual
    # receiver microbatch and compiled-counter behavior from the YAML/config.
    eval_logical_batch = int(get_cfg(cfg, "evaluation.logical_batch_size", 64))
    eval_microbatch = int(get_cfg(cfg, "evaluation.receiver_microbatch_size", min(16, eval_logical_batch)))
    set_cfg(cfg, "system.batch_size_eval", eval_logical_batch)
    set_cfg(cfg, "evaluation.receiver_microbatch_size", max(1, min(eval_microbatch, eval_logical_batch)))
    set_cfg(cfg, "evaluation.stream_eval_microbatches", bool(get_cfg(cfg, "evaluation.stream_eval_microbatches", True)))
    set_cfg(cfg, "evaluation.compiled_receiver_error_counts", bool(get_cfg(cfg, "evaluation.compiled_receiver_error_counts", True)))
    set_cfg(cfg, "evaluation.memory_cleanup_every_microbatch", bool(get_cfg(cfg, "evaluation.memory_cleanup_every_microbatch", True)))
    set_cfg(cfg, "evaluation.memory_cleanup_every_batches", int(get_cfg(cfg, "evaluation.memory_cleanup_every_batches", 1)))
    set_cfg(cfg, "evaluation.save_example_batch", variant_name == "main_d256_b4_r2" and num_users == 4)
    return cfg


def _checkpoint_path(cfg: dict[str, Any]) -> Path:
    output_root = PROJECT_ROOT / str(get_cfg(cfg, "experiment.output_root", "outputs"))
    name = str(get_cfg(cfg, "experiment.name", "experiment"))
    ckpt_name = str(get_cfg(cfg, "training.checkpoint_name", "best.weights.h5"))
    return output_root / name / "checkpoints" / ckpt_name


def _summary_path(cfg: dict[str, Any]) -> Path:
    output_root = PROJECT_ROOT / str(get_cfg(cfg, "experiment.output_root", "outputs"))
    name = str(get_cfg(cfg, "experiment.name", "experiment"))
    return output_root / name / "metrics" / "evaluation_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _release_tensorflow_state(label: str) -> None:
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        current = float(info.get("current", 0)) / (1024.0**3)
        peak = float(info.get("peak", 0)) / (1024.0**3)
        print(f"[COMPREHENSIVE] cleared TensorFlow state before {label}: gpu_mem={current:.2f}GiB peak={peak:.2f}GiB")
    except Exception:
        print(f"[COMPREHENSIVE] cleared TensorFlow state before {label}")


def _copy_curves(
    result: dict[str, Any],
    out_csv: Path,
    variant_name: str,
    label: str,
    dmrs_case: str,
    dmrs_label: str,
    num_users: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(result["curves_path"])
    df["dmrs_case"] = dmrs_case
    df["dmrs_label"] = dmrs_label
    df["variant"] = variant_name
    df["variant_label"] = label
    df["num_users"] = int(num_users)
    df["seed"] = int(seed)
    df["training_seed"] = int(seed)
    df["evaluation_seed"] = int(result.get("evaluation_seed", seed))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1--4 user comprehensive UPAIR ablation for the configured gNB antenna count.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--variants", default="main_d256_b4_r2,shallow_d256_b2_r2,deep_d256_b6_r2,narrow_d192_b4_r2,wide_d320_b4_r2,wide_deep_d320_b6_r2,mlpwide_d256_b4_r4")
    parser.add_argument("--dmrs-cases", default="1dmrs,2dmrs", help="Comma-separated DMRS cases to run. Default: 1dmrs,2dmrs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated random seeds. Defaults to system.seeds in the config.")
    parser.add_argument("--eval-users", default=None, help="Comma-separated evaluation user counts. Overrides multiuser.eval_num_users.")
    parser.add_argument("--use-optuna-best-1dmrs", action="store_true", help="Apply the best 1-DMRS Optuna hyperparameters for each architecture variant.")
    parser.add_argument("--optuna-best-storage-dir", default=str(PROJECT_ROOT / "optuna"), help="Directory containing <prefix>_<variant>.db Optuna files. Missing files fall back to built-in defaults unless --require-optuna-best is set.")
    parser.add_argument("--optuna-best-study-prefix", default="clean_b32_prb8_d256_u34610_1dmrs_stageC", help="Prefix for per-variant Optuna study and database names.")
    parser.add_argument("--require-optuna-best", action="store_true", help="Require external Optuna best JSON/DB instead of falling back to built-in recorded best parameters.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and reuse existing checkpoints.")
    parser.add_argument("--no-global-summary", action="store_true", help="Skip shared combined CSV/manifest writes. Use this for parallel Slurm array workers.")
    parser.add_argument("--force", action="store_true", help="Re-run training/evaluation even if resumable outputs already exist.")
    parser.add_argument("--plot", action="store_true", help="Generate TWC_plots_comprehensive figures after evaluation.")
    args = parser.parse_args()
    if args.plot and args.no_global_summary:
        raise ValueError("--plot requires global summary files; omit --no-global-summary or run merge_comprehensive_mu32_results.py first.")

    base_cfg = load_config(args.config)
    if args.eval_users:
        eval_users_override = [int(x.strip()) for x in args.eval_users.split(",") if x.strip()]
        if not eval_users_override:
            raise ValueError("--eval-users must include at least one integer user count.")
        set_cfg(base_cfg, "multiuser.eval_num_users", eval_users_override)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    dmrs_cases = [name.strip() for name in args.dmrs_cases.split(",") if name.strip()]
    seed_values = _parse_seed_values(base_cfg, args.seeds)
    eval_num_users = [int(x) for x in get_cfg(base_cfg, "multiuser.eval_num_users", [1, 2, 3, 4])]

    rx_tag = _rx_tag(base_cfg)
    csv_dir = PROJECT_ROOT / "TWC_plots_comprehensive" / f"csv_{rx_tag}"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "base_config": str(Path(args.config).resolve()),
        "num_rx_ant": int(get_cfg(base_cfg, "channel.num_rx_ant", 0)),
        "dmrs_cases": {},
        "eval_num_users": eval_num_users,
        "seeds": seed_values,
    }

    for seed in seed_values:
        for dmrs_case in dmrs_cases:
            if dmrs_case not in DMRS_CASES:
                raise KeyError(f"Unknown DMRS case {dmrs_case}. Available: {sorted(DMRS_CASES)}")
            dmrs_label = str(DMRS_CASES[dmrs_case]["label"])
            manifest["dmrs_cases"].setdefault(
                dmrs_case,
                {
                    "label": dmrs_label,
                    "overrides": DMRS_CASES[dmrs_case]["overrides"],
                    "variants": {},
                },
            )

            for variant_name in variant_names:
                train_cfg = _variant_cfg(base_cfg, variant_name, dmrs_case, seed)
                if args.use_optuna_best_1dmrs:
                    _apply_optuna_best_1dmrs(
                        train_cfg,
                        variant_name,
                        dmrs_case,
                        storage_dir=args.optuna_best_storage_dir,
                        study_prefix=args.optuna_best_study_prefix,
                        require_external=bool(args.require_optuna_best),
                    )
                label = str(VARIANTS[variant_name]["label"])

                if args.eval_only:
                    checkpoint_path = _checkpoint_path(train_cfg)
                    if not checkpoint_path.exists():
                        raise FileNotFoundError(f"--eval-only requested but checkpoint is missing: {checkpoint_path}")
                    train_result = {
                        "checkpoint_path": str(checkpoint_path),
                        "model_summary_path": str(_checkpoint_path(train_cfg).parents[1] / "metrics" / "model_summary.json"),
                        "training_complete": True,
                    }
                else:
                    if args.force:
                        set_cfg(train_cfg, "training.resume", False)
                    train_result = train_model(train_cfg)
                    if not bool(train_result.get("training_complete", True)):
                        raise SystemExit(
                            "Training stopped after saving resumable state. "
                            "Resubmit the same Slurm array task to continue from the saved checkpoint."
                        )
                    checkpoint_path = Path(train_result["checkpoint_path"])

                model_summary_path = Path(str(train_result.get("model_summary_path", "")))
                model_summary = _read_json(model_summary_path)
                variant_manifest = manifest["dmrs_cases"][dmrs_case]["variants"].setdefault(
                    variant_name,
                    {
                        "label": label,
                        "seed_runs": {},
                    },
                )
                variant_manifest["seed_runs"][str(seed)] = {
                    "training_seed": int(seed),
                    "evaluation_seed": int(get_cfg(train_cfg, "system.evaluation_seed", seed)),
                    "checkpoint_path": str(checkpoint_path),
                    "model_summary": model_summary,
                    "curves": {},
                }

                skip_final_eval = os.environ.get("UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL", "0").strip().lower() in {"1", "true", "yes", "y"}
                if skip_final_eval:
                    print(
                        f"[COMPREHENSIVE] skipping final evaluation by request "
                        f"UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1 for seed={seed} {dmrs_case}/{variant_name}. "
                        "Use the by-Eb/N0 eval phase after training completes."
                    )
                    continue

                for num_users in eval_num_users:
                    _release_tensorflow_state(f"evaluation seed={seed} {dmrs_case}/{variant_name}/u{num_users}")
                    cfg_eval = _eval_cfg(train_cfg, variant_name, dmrs_case, num_users)
                    out_csv = csv_dir / _seed_tag(seed) / dmrs_case / f"{variant_name}_u{num_users}_curves.csv"
                    summary_path = _summary_path(cfg_eval)
                    ignore_shared_eval_csv = os.environ.get("UPAIR_EVAL_IGNORE_SHARED_CSV", "0").strip().lower() in {"1", "true", "yes", "y"}
                    if out_csv.exists() and not args.force and not ignore_shared_eval_csv:
                        print(f"[COMPREHENSIVE] reusing existing evaluation CSV {out_csv}")
                        frame = pd.read_csv(out_csv)
                        if "seed" not in frame.columns:
                            frame["seed"] = int(seed)
                    else:
                        if args.force:
                            set_cfg(cfg_eval, "evaluation.force", True)
                        result = evaluate_model(cfg_eval, checkpoint_path=str(checkpoint_path), num_users=num_users)
                        if not bool(result.get("evaluation_complete", True)):
                            raise SystemExit(
                                "Evaluation stopped after saving resumable state. "
                                "Resubmit the same Slurm array task to continue from the saved evaluation state."
                            )
                        summary_path = Path(str(result["summary_path"]))
                        frame = _copy_curves(result, out_csv, variant_name, label, dmrs_case, dmrs_label, num_users, seed)
                    all_frames.append(frame)
                    variant_manifest["seed_runs"][str(seed)]["curves"][str(num_users)] = {
                        "csv": str(out_csv),
                        "summary": str(summary_path),
                    }

    if args.no_global_summary:
        print("[COMPREHENSIVE] skipped shared combined CSV/manifest writes for parallel worker mode")
    else:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = csv_dir / "comprehensive_curves.csv"
        combined.to_csv(combined_path, index=False)
        manifest["combined_csv"] = str(combined_path)
        if "dmrs_case" in combined.columns:
            for dmrs_case, case_df in combined.groupby("dmrs_case"):
                case_path = csv_dir / str(dmrs_case) / "comprehensive_curves.csv"
                case_path.parent.mkdir(parents=True, exist_ok=True)
                case_df.to_csv(case_path, index=False)
                if str(dmrs_case) in manifest["dmrs_cases"]:
                    manifest["dmrs_cases"][str(dmrs_case)]["combined_csv"] = str(case_path)

        manifest_path = csv_dir / "comprehensive_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        print(f"[COMPREHENSIVE] wrote {combined_path}")
        print(f"[COMPREHENSIVE] wrote {manifest_path}")

    if args.plot:
        from make_comprehensive_mu32_plots import main as plot_main

        plot_main()


if __name__ == "__main__":
    main()
