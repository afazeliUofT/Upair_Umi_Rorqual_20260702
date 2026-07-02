from __future__ import annotations

import gc
import json
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf

from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, extract_true_dmrs_mask_per_stream, get_resource_grid, max_num_users, multiuser_enabled
from .config import ensure_output_tree, get_cfg
from .estimator import UPAIRChannelEstimator
from .impairments import apply_rf_impairments_to_transmit_grid_if_enabled, apply_symbol_phase_impairment
from .utils import (
    call_channel,
    call_transmitter,
    complex_sq_abs,
    compute_nmse,
    ebno_db_to_no,
    save_json,
    save_yaml,
    set_global_seed,
    tf_float,
)


def _make_batch(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    batch_size: int,
    training: bool,
    fixed_ebno_db: float | None = None,
) -> dict[str, tf.Tensor]:
    x, _ = call_transmitter(tx, batch_size)
    x, rf_meta = apply_rf_impairments_to_transmit_grid_if_enabled(x, tx, cfg, training=training)

    if fixed_ebno_db is None:
        ebno_db = tf.random.uniform(
            [],
            minval=float(get_cfg(cfg, "system.ebno_db_train_min", 0.0)),
            maxval=float(get_cfg(cfg, "system.ebno_db_train_max", 16.0)),
            dtype=tf.float32,
        )
    else:
        ebno_db = tf.constant(float(fixed_ebno_db), tf.float32)

    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=training)

    return {
        "y": y,
        "h": h,
        "no": no,
        "ebno_db": ebno_db,
        "rf_mode": rf_meta.get("rf_mode", "clean"),
    }


def _sample_train_num_users(cfg: dict[str, Any]) -> int:
    max_users = max_num_users(cfg)
    if max_users <= 1:
        return 1
    sampler = str(get_cfg(cfg, "multiuser.train_user_count_sampler", "weighted")).lower()
    users = np.arange(1, max_users + 1)
    if sampler == "uniform":
        weights = np.ones_like(users, dtype=np.float64)
    elif sampler in {"fixed_max", "max"}:
        return int(max_users)
    elif sampler in {"weighted", "custom"}:
        configured = get_cfg(cfg, "multiuser.train_user_count_weights", None)
        if configured is None:
            weights = users.astype(np.float64)
        else:
            weights = np.asarray([float(x) for x in configured], dtype=np.float64)
            if weights.shape[0] != max_users:
                raise ValueError(
                    "multiuser.train_user_count_weights must contain exactly "
                    f"{max_users} entries, got {weights.shape[0]}."
                )
    elif sampler in {"triangular", "linear"}:
        # Linear weighting: P(U=max)>...>P(U=1).
        weights = users.astype(np.float64)
    elif sampler in {"quadratic", "power2"}:
        weights = np.square(users.astype(np.float64))
    else:
        raise ValueError(
            "Unknown multiuser.train_user_count_sampler="
            f"{sampler!r}. Use uniform, weighted, triangular, quadratic, or fixed_max."
        )
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        raise ValueError("multiuser user-count sampling weights must have a positive sum.")
    probs = weights / weights.sum()
    return int(np.random.choice(users, p=probs))


def _build_system_for_num_users(cfg: dict[str, Any], num_users: int) -> dict[str, Any]:
    tx, _ = build_pusch_transmitter(cfg, num_users=num_users)
    channel = build_channel(cfg, tx)
    ls_estimator = build_ls_estimator(tx, cfg)
    resource_grid = get_resource_grid(tx)
    return {
        "num_users": num_users,
        "tx": tx,
        "channel": channel,
        "ls_estimator": ls_estimator,
        "resource_grid": resource_grid,
        "pilot_mask": extract_true_dmrs_mask_per_stream(tx, resource_grid),
    }


def _build_training_systems(cfg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not multiuser_enabled(cfg):
        return {1: _build_system_for_num_users(cfg, 1)}
    return {
        num_users: _build_system_for_num_users(cfg, num_users)
        for num_users in range(1, max_num_users(cfg) + 1)
    }


def _learning_rate_summary(cfg: dict[str, Any]) -> dict[str, float | str]:
    return {
        "schedule": str(get_cfg(cfg, "training.learning_rate_schedule", "constant")).lower(),
        "learning_rate": float(get_cfg(cfg, "training.learning_rate", 3e-4)),
        "final_fraction": float(get_cfg(cfg, "training.learning_rate_final_fraction", 0.05)),
        "decay_rate": float(get_cfg(cfg, "training.learning_rate_decay_rate", 0.96)),
        "decay_steps": float(get_cfg(cfg, "training.learning_rate_decay_steps", get_cfg(cfg, "training.steps", 1))),
        "polynomial_power": float(get_cfg(cfg, "training.learning_rate_polynomial_power", 1.0)),
        "restart_first_decay_steps": float(
            get_cfg(cfg, "training.learning_rate_restart_first_decay_steps", max(1, int(get_cfg(cfg, "training.steps", 1)) // 4))
        ),
        "restart_t_mul": float(get_cfg(cfg, "training.learning_rate_restart_t_mul", 2.0)),
        "restart_m_mul": float(get_cfg(cfg, "training.learning_rate_restart_m_mul", 0.8)),
    }


def _make_learning_rate(cfg: dict[str, Any]) -> float | tf.keras.optimizers.schedules.LearningRateSchedule:
    base_lr = float(get_cfg(cfg, "training.learning_rate", 3e-4))
    total_steps = max(1, int(get_cfg(cfg, "training.steps", 1)))
    schedule = str(get_cfg(cfg, "training.learning_rate_schedule", "constant")).lower()
    if schedule in {"constant", "fixed", "none"}:
        return base_lr

    final_fraction = max(0.0, min(1.0, float(get_cfg(cfg, "training.learning_rate_final_fraction", 0.05))))
    decay_steps = max(1, int(get_cfg(cfg, "training.learning_rate_decay_steps", total_steps)))
    if schedule in {"cosine", "cosine_decay"}:
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=decay_steps,
            alpha=final_fraction,
        )
    if schedule in {"polynomial", "polynomial_decay"}:
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=base_lr,
            decay_steps=decay_steps,
            end_learning_rate=base_lr * final_fraction,
            power=float(get_cfg(cfg, "training.learning_rate_polynomial_power", 1.0)),
        )
    if schedule in {"exponential", "exponential_decay"}:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr,
            decay_steps=decay_steps,
            decay_rate=float(get_cfg(cfg, "training.learning_rate_decay_rate", 0.96)),
            staircase=False,
        )
    if schedule in {"cosine_restarts", "cosine_decay_restarts"}:
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_lr,
            first_decay_steps=max(
                1,
                int(get_cfg(cfg, "training.learning_rate_restart_first_decay_steps", max(1, total_steps // 4))),
            ),
            t_mul=float(get_cfg(cfg, "training.learning_rate_restart_t_mul", 2.0)),
            m_mul=float(get_cfg(cfg, "training.learning_rate_restart_m_mul", 0.8)),
            alpha=final_fraction,
        )

    raise ValueError(f"Unknown training.learning_rate_schedule={schedule!r}.")


def _make_optimizer(cfg: dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    lr = _make_learning_rate(cfg)
    wd = float(cfg["training"]["weight_decay"])
    optimizer_jit_compile = bool(get_cfg(cfg, "training.optimizer_jit_compile", False))
    adamw_kwargs = {
        "learning_rate": lr,
        "weight_decay": wd,
    }
    if optimizer_jit_compile:
        adamw_kwargs["jit_compile"] = True
    try:
        return tf.keras.optimizers.AdamW(**adamw_kwargs)
    except (TypeError, ValueError):
        adamw_kwargs.pop("jit_compile", None)
        try:
            return tf.keras.optimizers.AdamW(**adamw_kwargs)
        except Exception:
            pass
    except Exception:
        pass

    adam_kwargs = {"learning_rate": lr}
    if optimizer_jit_compile:
        adam_kwargs["jit_compile"] = True
    try:
        return tf.keras.optimizers.Adam(**adam_kwargs)
    except (TypeError, ValueError):
        return tf.keras.optimizers.Adam(learning_rate=lr)


def _current_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    for attr in ("learning_rate", "_learning_rate", "lr"):
        lr = getattr(optimizer, attr, None)
        if lr is None:
            continue
        try:
            value = lr(optimizer.iterations) if callable(lr) else lr
            return float(tf.convert_to_tensor(value).numpy())
        except Exception:
            continue
    return float("nan")


@tf.function(reduce_retracing=True)
def _train_step(
    estimator: UPAIRChannelEstimator,
    optimizer: tf.keras.optimizers.Optimizer,
    y: tf.Tensor,
    h: tf.Tensor,
    no: tf.Tensor,
    nmse_loss_weight: float,
    grad_clip_norm: float,
    ls_estimator: Any | None = None,
    pilot_mask: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    with tf.GradientTape() as tape:
        h_hat, err_hat, h_ls, _ = estimator.estimate_with_ls(
            y,
            no,
            training=True,
            ls_estimator=ls_estimator,
            pilot_mask=pilot_mask,
        )
        target = tf.convert_to_tensor(h)

        residual = target - h_hat
        residual_ls = target - h_ls
        sq_err = complex_sq_abs(residual)
        power = tf.reduce_mean(complex_sq_abs(target)) + 1e-9

        loss_nll = tf.reduce_mean(sq_err / (err_hat + 1e-6) + tf.math.log(err_hat + 1e-6))
        nmse_prop = tf.reduce_mean(sq_err) / power
        nmse_ls = tf.reduce_mean(complex_sq_abs(residual_ls)) / power
        loss = loss_nll + float(nmse_loss_weight) * nmse_prop

    grads = tape.gradient(loss, estimator.trainable_variables)
    grad_var_pairs = [(g, v) for g, v in zip(grads, estimator.trainable_variables) if g is not None]
    if grad_var_pairs:
        grad_tensors = [g for g, _ in grad_var_pairs]
        clipped_grads, _ = tf.clip_by_global_norm(grad_tensors, float(grad_clip_norm))
        optimizer.apply_gradients(zip(clipped_grads, [v for _, v in grad_var_pairs]))

    return {
        "loss": tf.cast(loss, tf.float32),
        "loss_nll": tf.cast(loss_nll, tf.float32),
        "nmse_prop": tf.cast(nmse_prop, tf.float32),
        "nmse_ls": tf.cast(nmse_ls, tf.float32),
    }


@tf.function(reduce_retracing=True)
def _validation_step(
    estimator: UPAIRChannelEstimator,
    y: tf.Tensor,
    h: tf.Tensor,
    no: tf.Tensor,
    ls_estimator: Any | None = None,
    pilot_mask: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    h_hat, _, h_ls, _ = estimator.estimate_with_ls(
        y,
        no,
        training=False,
        ls_estimator=ls_estimator,
        pilot_mask=pilot_mask,
    )
    return {
        "nmse_prop": compute_nmse(h, h_hat),
        "nmse_ls": compute_nmse(h, h_ls),
    }


def _release_training_memory() -> None:
    async_wait = getattr(getattr(tf, "experimental", object()), "async_wait", None)
    if callable(async_wait):
        try:
            async_wait()
        except Exception:
            pass
    gc.collect()


def _gpu_memory_message() -> str:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except Exception:
        return ""
    current_gib = float(info.get("current", 0)) / (1024.0**3)
    peak_gib = float(info.get("peak", 0)) / (1024.0**3)
    return f" gpu_mem={current_gib:.2f}GiB peak={peak_gib:.2f}GiB"


def _logical_microbatch_sizes(total_batch_size: int, microbatch_size: int) -> list[int]:
    total_batch_size = max(1, int(total_batch_size))
    microbatch_size = max(1, min(int(microbatch_size), total_batch_size))
    sizes: list[int] = []
    remaining = total_batch_size
    while remaining > 0:
        current = min(microbatch_size, remaining)
        sizes.append(current)
        remaining -= current
    return sizes


def _validate(
    estimator: UPAIRChannelEstimator,
    systems: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    val_steps = int(cfg["training"]["val_steps"])
    eval_grid = list(get_cfg(cfg, "system.ebno_db_eval", [10]))
    val_ebno_cfg = get_cfg(cfg, "training.val_ebno_db", None)
    if val_ebno_cfg is None:
        ebno_values = [float(eval_grid[min(len(eval_grid) // 2, len(eval_grid) - 1)])]
    elif isinstance(val_ebno_cfg, (int, float)):
        ebno_values = [float(val_ebno_cfg)]
    else:
        ebno_values = [float(x) for x in val_ebno_cfg]
    if not ebno_values:
        raise ValueError("training.val_ebno_db must contain at least one Eb/N0 value.")
    val_snr_sampling_mode = str(get_cfg(cfg, "training.val_snr_sampling_mode", "sampled_grid")).lower()

    val_sampling_mode = str(get_cfg(cfg, "training.val_sampling_mode", "sampled")).lower()
    val_user_counts_cfg = get_cfg(cfg, "training.val_user_counts", list(sorted(systems)))
    val_user_counts = [int(x) for x in val_user_counts_cfg if int(x) in systems]
    if not val_user_counts:
        val_user_counts = list(sorted(systems))
    val_user_weights_cfg = get_cfg(cfg, "training.val_user_count_weights", None)
    if val_user_weights_cfg is None:
        val_user_weights = np.asarray(val_user_counts, dtype=np.float64)
    else:
        raw_weights = [float(x) for x in val_user_weights_cfg]
        if len(raw_weights) != len([int(x) for x in val_user_counts_cfg]):
            raise ValueError("training.val_user_count_weights must match training.val_user_counts length.")
        weight_by_user = {int(user): float(weight) for user, weight in zip(val_user_counts_cfg, raw_weights, strict=False)}
        val_user_weights = np.asarray([weight_by_user[int(user)] for user in val_user_counts], dtype=np.float64)
    val_user_weights = np.maximum(val_user_weights, 0.0)
    if float(val_user_weights.sum()) <= 0.0:
        raise ValueError("training.val_user_count_weights must have a positive sum over available users.")
    val_user_probs = val_user_weights / val_user_weights.sum()

    logical_eval_batch_size = int(cfg["system"]["batch_size_eval"])
    val_microbatch_size = int(get_cfg(cfg, "training.val_microbatch_size", logical_eval_batch_size))
    if val_microbatch_size <= 0:
        val_microbatch_size = logical_eval_batch_size
    microbatch_sizes = _logical_microbatch_sizes(logical_eval_batch_size, val_microbatch_size)
    cleanup_every_microbatch = bool(get_cfg(cfg, "training.val_memory_cleanup_every_microbatch", False))

    nmse_prop = []
    nmse_ls = []
    user_count_hist = {str(user): 0 for user in sorted(systems)}
    snr_hist = {str(float(value)): 0 for value in ebno_values}

    for idx in range(val_steps):
        if val_snr_sampling_mode in {"cycle", "cyclic", "grid_cycle"}:
            ebno_for_val = float(ebno_values[idx % len(ebno_values)])
        elif val_snr_sampling_mode in {"uniform_train_range", "train_uniform"}:
            ebno_for_val = float(
                np.random.uniform(
                    float(get_cfg(cfg, "system.ebno_db_train_min", min(ebno_values))),
                    float(get_cfg(cfg, "system.ebno_db_train_max", max(ebno_values))),
                )
            )
        else:
            ebno_for_val = float(np.random.choice(np.asarray(ebno_values, dtype=np.float64)))

        if val_sampling_mode in {"sampled", "weighted", "weighted_sampled", "random"}:
            selected_users = int(np.random.choice(np.asarray(val_user_counts, dtype=np.int64), p=val_user_probs))
        else:
            selected_users = _sample_train_num_users(cfg)
        system = systems[selected_users]
        user_count_hist[str(int(system["num_users"]))] = user_count_hist.get(str(int(system["num_users"])), 0) + 1
        snr_hist[str(float(ebno_for_val))] = snr_hist.get(str(float(ebno_for_val)), 0) + 1

        prop_weighted_sum = 0.0
        ls_weighted_sum = 0.0
        weight_sum = 0.0
        for current_batch_size in microbatch_sizes:
            batch = _make_batch(
                tx=system["tx"],
                channel=system["channel"],
                cfg=cfg,
                batch_size=int(current_batch_size),
                training=False,
                fixed_ebno_db=ebno_for_val,
            )
            metrics = _validation_step(
                estimator=estimator,
                y=batch["y"],
                h=batch["h"],
                no=batch["no"],
                ls_estimator=system["ls_estimator"],
                pilot_mask=system["pilot_mask"],
            )
            current_weight = float(current_batch_size)
            prop_weighted_sum += float(metrics["nmse_prop"].numpy()) * current_weight
            ls_weighted_sum += float(metrics["nmse_ls"].numpy()) * current_weight
            weight_sum += current_weight
            del metrics, batch
            if cleanup_every_microbatch:
                _release_training_memory()
        nmse_prop.append(prop_weighted_sum / max(weight_sum, 1.0))
        nmse_ls.append(ls_weighted_sum / max(weight_sum, 1.0))

    _release_training_memory()
    return {
        "val_nmse_prop": float(np.mean(nmse_prop)),
        "val_nmse_ls": float(np.mean(nmse_ls)),
        "val_ebno_db": ebno_values[0] if len(ebno_values) == 1 else "multi",
        "val_ebno_db_values": ebno_values,
        "val_batch_size": int(logical_eval_batch_size),
        "val_microbatch_size": int(min(val_microbatch_size, logical_eval_batch_size)),
        "val_user_count_hist": user_count_hist,
        "val_snr_hist": snr_hist,
    }


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("history", [])
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    except Exception as exc:
        print(f"[TRAIN] ignoring unreadable history file {path}: {exc!r}")
    return []


def _warmup_estimator(
    estimator: UPAIRChannelEstimator,
    system: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    eval_grid = list(get_cfg(cfg, "system.ebno_db_eval", [10]))
    batch = _make_batch(
        tx=system["tx"],
        channel=system["channel"],
        cfg=cfg,
        batch_size=max(1, min(2, int(cfg["system"]["batch_size_train"]))),
        training=False,
        fixed_ebno_db=float(eval_grid[min(len(eval_grid) // 2, len(eval_grid) - 1)]),
    )
    estimator.estimate_with_ls(
        batch["y"],
        batch["no"],
        training=False,
        ls_estimator=system["ls_estimator"],
        pilot_mask=system["pilot_mask"],
    )
    del batch
    _release_training_memory()


def train_model(
    cfg: dict[str, Any],
    validation_callback: Callable[[dict[str, Any], int], None] | None = None,
) -> dict[str, Any]:
    set_global_seed(int(cfg["system"]["seed"]))
    if bool(get_cfg(cfg, "system.graph_mode", True)):
        tf.config.run_functions_eagerly(False)
    paths = ensure_output_tree(cfg)

    systems = _build_training_systems(cfg)
    reference_system = systems[max(systems)]
    estimator = UPAIRChannelEstimator(
        ls_estimator=reference_system["ls_estimator"],
        resource_grid=reference_system["resource_grid"],
        cfg=cfg,
        pilot_mask=reference_system["pilot_mask"],
    )
    optimizer = _make_optimizer(cfg)

    ckpt_path = paths["checkpoints"] / str(cfg["training"]["checkpoint_name"])
    history_path = paths["metrics"] / "history.json"
    train_state_path = paths["metrics"] / "train_state.json"
    state_dir = paths["checkpoints"] / "training_state"

    _warmup_estimator(estimator, reference_system, cfg)
    try:
        optimizer.build(estimator.trainable_variables)
    except Exception:
        pass

    step_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="training_step")
    best_val_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False, name="best_val")
    last_loss_var = tf.Variable(np.nan, dtype=tf.float32, trainable=False, name="last_loss")
    last_nmse_prop_var = tf.Variable(np.nan, dtype=tf.float32, trainable=False, name="last_nmse_prop")
    last_nmse_ls_var = tf.Variable(np.nan, dtype=tf.float32, trainable=False, name="last_nmse_ls")
    training_ckpt = tf.train.Checkpoint(
        step=step_var,
        best_val=best_val_var,
        last_loss=last_loss_var,
        last_nmse_prop=last_nmse_prop_var,
        last_nmse_ls=last_nmse_ls_var,
        optimizer=optimizer,
        estimator=estimator,
    )
    manager = tf.train.CheckpointManager(
        training_ckpt,
        directory=str(state_dir),
        max_to_keep=int(get_cfg(cfg, "training.max_resume_checkpoints", 3)),
        checkpoint_name="ckpt",
    )

    total_steps = int(cfg["training"]["steps"])
    log_every = int(cfg["training"]["log_every"])
    eval_every = int(cfg["training"]["eval_every"])
    checkpoint_every = int(get_cfg(cfg, "training.checkpoint_every", log_every))
    nmse_loss_weight = float(cfg["training"]["nmse_loss_weight"])
    grad_clip_norm = float(cfg["training"]["grad_clip_norm"])
    memory_cleanup_every_steps = int(get_cfg(cfg, "training.memory_cleanup_every_steps", 0))
    resume_enabled = bool(get_cfg(cfg, "training.resume", True))
    lr_summary = _learning_rate_summary(cfg)
    training_seed = int(get_cfg(cfg, "system.training_seed", cfg["system"]["seed"]))
    evaluation_seed = int(get_cfg(cfg, "system.evaluation_seed", cfg["system"]["seed"]))
    print(
        "[TRAIN] optimizer "
        f"lr_schedule={lr_summary['schedule']} "
        f"learning_rate={lr_summary['learning_rate']:.6g} "
        f"weight_decay={float(cfg['training']['weight_decay']):.6g}"
    )
    print(
        "[TRAIN] seeds "
        f"active_seed={int(cfg['system']['seed'])} "
        f"training_seed={training_seed} "
        f"evaluation_seed={evaluation_seed}"
    )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    start_step = 1
    last_metrics: dict[str, float] | None = None

    if resume_enabled and manager.latest_checkpoint:
        status = training_ckpt.restore(manager.latest_checkpoint)
        status.expect_partial()
        start_step = int(step_var.numpy()) + 1
        best_val = float(best_val_var.numpy())
        history = _load_history(history_path)
        if history:
            last_metrics = dict(history[-1])
        else:
            last_metrics = {
                "loss": float(last_loss_var.numpy()),
                "nmse_prop": float(last_nmse_prop_var.numpy()),
                "nmse_ls": float(last_nmse_ls_var.numpy()),
            }
        print(
            f"[TRAIN] resumed from {manager.latest_checkpoint} "
            f"at completed_step={start_step - 1} best_val={best_val:.6g}"
        )
    else:
        print("[TRAIN] starting from scratch")

    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[TRAIN] received signal {signum}; will save and stop after current step.")

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_stop)
        except Exception:
            pass

    def _save_progress(step: int, reason: str, complete: bool = False) -> str:
        step_var.assign(int(step))
        best_val_var.assign(float(best_val))
        latest_checkpoint = manager.save(checkpoint_number=int(step))
        save_json({"history": history}, history_path)
        save_json(
            {
                "latest_step": int(step),
                "total_steps": int(total_steps),
                "training_complete": bool(complete),
                "best_val": float(best_val),
                "best_weights_path": str(ckpt_path),
                "latest_training_state_checkpoint": str(latest_checkpoint),
                "resume_enabled": bool(resume_enabled),
                "save_reason": reason,
                "last_metrics": last_metrics,
                "checkpoint_metrics": {
                    "last_loss": float(last_loss_var.numpy()),
                    "last_nmse_prop": float(last_nmse_prop_var.numpy()),
                    "last_nmse_ls": float(last_nmse_ls_var.numpy()),
                },
                "training_parameters": {
                    "batch_size_train": int(cfg["system"]["batch_size_train"]),
                    "batch_size_eval": int(cfg["system"]["batch_size_eval"]),
                    "learning_rate": float(cfg["training"]["learning_rate"]),
                    "current_learning_rate": float(_current_learning_rate(optimizer)),
                    "learning_rate_schedule": lr_summary,
                    "weight_decay": float(cfg["training"]["weight_decay"]),
                    "nmse_loss_weight": float(nmse_loss_weight),
                    "grad_clip_norm": float(grad_clip_norm),
                    "eval_every": int(eval_every),
                    "val_steps": int(cfg["training"]["val_steps"]),
                    "val_microbatch_size": int(get_cfg(cfg, "training.val_microbatch_size", cfg["system"]["batch_size_eval"])),
                    "memory_cleanup_every_steps": int(memory_cleanup_every_steps),
                    "seed": int(cfg["system"]["seed"]),
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                },
            },
            train_state_path,
        )
        print(f"[TRAIN] saved resumable state at step={step} reason={reason}: {latest_checkpoint}")
        return latest_checkpoint

    current_step = start_step - 1
    last_completed_step = start_step - 1
    training_complete = last_completed_step >= total_steps
    training_start_s = time.perf_counter()

    try:
        for step in range(start_step, total_steps + 1):
            step_start_s = time.perf_counter()
            current_step = step
            system = systems[_sample_train_num_users(cfg)]
            batch = _make_batch(
                tx=system["tx"],
                channel=system["channel"],
                cfg=cfg,
                batch_size=int(cfg["system"]["batch_size_train"]),
                training=True,
            )
            metrics = _train_step(
                estimator=estimator,
                optimizer=optimizer,
                y=batch["y"],
                h=batch["h"],
                no=batch["no"],
                nmse_loss_weight=nmse_loss_weight,
                grad_clip_norm=grad_clip_norm,
                ls_estimator=system["ls_estimator"],
                pilot_mask=system["pilot_mask"],
            )
            row = {
                "step": step,
                "num_users": int(system["num_users"]),
                "ebno_db": float(batch["ebno_db"].numpy()),
                "loss": float(metrics["loss"].numpy()),
                "loss_nll": float(metrics["loss_nll"].numpy()),
                "nmse_prop": float(metrics["nmse_prop"].numpy()),
                "nmse_ls": float(metrics["nmse_ls"].numpy()),
                "learning_rate": float(_current_learning_rate(optimizer)),
            }
            del metrics, batch

            did_validate = False
            if step % eval_every == 0 or step == total_steps:
                did_validate = True
                val_metrics = _validate(estimator, systems, cfg)
                row.update(val_metrics)
                if val_metrics["val_nmse_prop"] < best_val:
                    best_val = val_metrics["val_nmse_prop"]
                    estimator.save_weights(str(ckpt_path))
                    print(f"[TRAIN] saved new best weights at step={step} val_nmse={best_val:.6g}")
                del val_metrics

            history.append(row)
            row["step_elapsed_s"] = float(time.perf_counter() - step_start_s)
            row["training_elapsed_s"] = float(time.perf_counter() - training_start_s)
            row["steps_per_s"] = float((step - start_step + 1) / max(row["training_elapsed_s"], 1e-9))
            last_metrics = dict(row)
            last_loss_var.assign(float(row["loss"]))
            last_nmse_prop_var.assign(float(row["nmse_prop"]))
            last_nmse_ls_var.assign(float(row["nmse_ls"]))
            last_completed_step = step
            current_step = last_completed_step

            if step % log_every == 0 or step == 1 or step == total_steps:
                print(
                    f"[TRAIN] step={step:05d} "
                    f"loss={row['loss']:.5f} "
                    f"nmse_prop={row['nmse_prop']:.5f} "
                    f"nmse_ls={row['nmse_ls']:.5f}"
                    f" lr={row['learning_rate']:.3e}"
                    f" step_time={row['step_elapsed_s']:.3f}s "
                    f"steps_per_s={row['steps_per_s']:.3f}"
                    f"{_gpu_memory_message()}"
                )

            if step % checkpoint_every == 0 or did_validate or step == total_steps or stop_requested:
                _save_progress(step, "periodic" if not stop_requested else "signal", complete=step >= total_steps)

            if did_validate and validation_callback is not None and not stop_requested:
                validation_callback(dict(row), int(step))

            cleanup_every = int(get_cfg(cfg, "training.memory_cleanup_every_steps", 0))
            if bool(get_cfg(cfg, "training.memory_cleanup_after_validation", True)) and did_validate:
                _release_training_memory()
            elif cleanup_every > 0 and step % cleanup_every == 0:
                _release_training_memory()

            if stop_requested:
                print("[TRAIN] stopped early after saving resumable state; resubmit the Slurm task to continue.")
                break
        training_complete = last_completed_step >= total_steps and not stop_requested
    except KeyboardInterrupt:
        _save_progress(last_completed_step, "keyboard_interrupt", complete=False)
        raise
    except tf.errors.ResourceExhaustedError:
        _save_progress(last_completed_step, "resource_exhausted", complete=False)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    if not ckpt_path.exists():
        estimator.save_weights(str(ckpt_path))

    _save_progress(last_completed_step, "final" if training_complete else "incomplete", complete=training_complete)
    save_json(
        {
            "num_trainable_params": int(np.sum([np.prod(v.shape) for v in estimator.trainable_variables])),
            "multiuser_enabled": bool(multiuser_enabled(cfg)),
            "max_num_users": int(max_num_users(cfg)),
            "train_user_count_sampler": str(get_cfg(cfg, "multiuser.train_user_count_sampler", "weighted")),
            "train_user_count_weights": get_cfg(cfg, "multiuser.train_user_count_weights", None),
            "training_complete": bool(training_complete),
            "latest_step": int(last_completed_step),
            "total_steps": int(total_steps),
            "best_val": float(best_val),
            "batch_size_train": int(cfg["system"]["batch_size_train"]),
            "batch_size_eval": int(cfg["system"]["batch_size_eval"]),
            "val_microbatch_size": int(get_cfg(cfg, "training.val_microbatch_size", cfg["system"]["batch_size_eval"])),
            "memory_cleanup_every_steps": int(memory_cleanup_every_steps),
            "learning_rate": float(cfg["training"]["learning_rate"]),
            "current_learning_rate": float(_current_learning_rate(optimizer)),
            "learning_rate_schedule": lr_summary,
            "weight_decay": float(cfg["training"]["weight_decay"]),
            "seed": int(cfg["system"]["seed"]),
        },
        paths["metrics"] / "model_summary.json",
    )
    save_yaml(cfg, paths["artifacts"] / "resolved_config.yaml")

    return {
        "output_dir": str(paths["root"]),
        "checkpoint_path": str(ckpt_path),
        "history_path": str(history_path),
        "model_summary_path": str(paths["metrics"] / "model_summary.json"),
        "train_state_path": str(train_state_path),
        "training_complete": bool(training_complete),
        "latest_step": int(last_completed_step),
        "total_steps": int(total_steps),
    }
