from __future__ import annotations

import gc
import math
import signal
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .baselines import (
    PERFECT_RECEIVER,
    PROPOSED_RECEIVER,
    build_classical_baseline_suite,
    classical_receivers_from_cfg,
    enabled_receivers_from_cfg,
    wants_receiver,
)
from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, build_receiver, extract_true_dmrs_mask_per_stream, get_resource_grid, max_num_users, multiuser_enabled
from .compat import safe_call_variants
from .config import ensure_output_tree, get_cfg
from .estimator import UPAIRChannelEstimator
from .impairments import apply_rf_impairments_to_transmit_grid_if_enabled, apply_symbol_phase_impairment
from .utils import (
    call_channel,
    call_transmitter,
    compute_ber,
    compute_bler_from_crc,
    complex_sq_abs,
    ebno_db_to_no,
    flatten_bits,
    infer_receiver_output,
    save_json,
    save_yaml,
    set_global_seed,
)


def _call_channel_estimator(estimator: Any, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    try:
        out = estimator(y, no)
    except (tf.errors.ResourceExhaustedError, MemoryError):
        raise
    except Exception:
        out = safe_call_variants(estimator, y, no)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("Channel estimator must return (h_hat, err_var).")
    return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])


def _make_eval_batch(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    batch_size: int,
    ebno_db: float,
) -> dict[str, tf.Tensor]:
    x, bits = call_transmitter(tx, batch_size)
    x, rf_meta = apply_rf_impairments_to_transmit_grid_if_enabled(x, tx, cfg, training=False)
    no = ebno_db_to_no(tf.constant(float(ebno_db), tf.float32), tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=False)
    return {"b": bits, "y": y, "h": h, "no": no, "ebno_db": tf.constant(float(ebno_db), tf.float32), "rf_mode": rf_meta.get("rf_mode", "clean")}


def _first_dim(value: tf.Tensor) -> int | None:
    tensor = tf.convert_to_tensor(value)
    if tensor.shape.rank == 0:
        return None
    if tensor.shape[0] is not None:
        return int(tensor.shape[0])
    try:
        return int(tf.shape(tensor)[0].numpy())
    except Exception:
        return None


def _slice_batch_axis(value: tf.Tensor | None, start: int, end: int, batch_size: int) -> tf.Tensor | None:
    if value is None:
        return None
    tensor = tf.convert_to_tensor(value)
    if _first_dim(tensor) == int(batch_size):
        return tensor[start:end]
    return tensor


def _iter_eval_microbatches(batch: dict[str, tf.Tensor], microbatch_size: int) -> Iterator[dict[str, tf.Tensor]]:
    batch_size = int(tf.shape(batch["y"])[0].numpy())
    microbatch_size = max(1, min(int(microbatch_size), batch_size))
    for start in range(0, batch_size, microbatch_size):
        end = min(start + microbatch_size, batch_size)
        yield {
            key: _slice_batch_axis(value, start, end, batch_size)  # type: ignore[arg-type]
            for key, value in batch.items()
        }


def _release_eval_memory() -> None:
    async_wait = getattr(getattr(tf, "experimental", object()), "async_wait", None)
    if callable(async_wait):
        try:
            async_wait()
        except Exception:
            pass
    gc.collect()


def _gpu_memory_stats() -> dict[str, float]:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except Exception:
        return {"gpu_mem_gib": float("nan"), "gpu_peak_gib": float("nan")}
    return {
        "gpu_mem_gib": float(info.get("current", 0)) / (1024.0**3),
        "gpu_peak_gib": float(info.get("peak", 0)) / (1024.0**3),
    }


def _gpu_memory_message() -> str:
    stats = _gpu_memory_stats()
    if np.isnan(stats["gpu_mem_gib"]):
        return ""
    return f" gpu_mem={stats['gpu_mem_gib']:.2f}GiB peak={stats['gpu_peak_gib']:.2f}GiB"


def _nmse_components(h_true: tf.Tensor, h_hat: tf.Tensor) -> tuple[float, float]:
    numerator = tf.reduce_sum(complex_sq_abs(tf.convert_to_tensor(h_true) - tf.convert_to_tensor(h_hat)))
    denominator = tf.reduce_sum(complex_sq_abs(h_true))
    return float(numerator.numpy()), float(denominator.numpy())


def _safe_concat(parts: list[tf.Tensor]) -> tf.Tensor | None:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return tf.concat(parts, axis=0)


def _metric_min(df: pd.DataFrame, receiver: str, metric: str) -> float | None:
    sub = df[df["receiver"] == receiver][metric].dropna()
    if sub.empty:
        return None
    return float(sub.min())


def _best_classical_row(df: pd.DataFrame, metric: str, reliable_only: bool = False) -> dict[str, float | str] | None:
    sub = df[["receiver", metric]].dropna().copy()
    if reliable_only and metric in {"ber", "bler"}:
        reliability_col = f"reliable_{metric}"
        if reliability_col in df.columns:
            sub = df.loc[df[reliability_col].fillna(False), ["receiver", metric]].dropna().copy()
    if sub.empty:
        return None
    idx = sub[metric].idxmin()
    row = sub.loc[idx]
    return {"receiver": str(row["receiver"]), "value": float(row[metric])}


def _build_summary(
    df: pd.DataFrame,
    checkpoint_path: str | None,
    enabled_receivers: list[str],
    artifacts: dict[str, str],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    classical_receivers = classical_receivers_from_cfg({"baselines": {"enabled_receivers": enabled_receivers}})
    classical_df = df[df["receiver"].isin(classical_receivers)].copy()

    summary: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "enabled_receivers": enabled_receivers,
        "classical_receivers": classical_receivers,
        "num_users": int(df["num_users"].dropna().iloc[0]) if "num_users" in df.columns and not df["num_users"].dropna().empty else None,
        "num_curve_rows": int(len(df)),
        "evaluation_controls": {
            "min_num_batches_per_point": int(eval_cfg["min_num_batches_per_point"]),
            "max_num_batches_per_point": int(eval_cfg["max_num_batches_per_point"]),
            "target_block_errors_per_receiver": int(eval_cfg["target_block_errors_per_receiver"]),
            "per_receiver_stopping": bool(eval_cfg["per_receiver_stopping"]),
            "target_bler_floor": float(eval_cfg["target_bler_floor"]),
            "floor_confidence_factor": float(eval_cfg["floor_confidence_factor"]),
            "floor_num_frames": int(eval_cfg["floor_num_frames"]),
            "reliable_min_block_errors": int(eval_cfg["reliable_min_block_errors"]),
            "reliable_min_bit_errors": int(eval_cfg["reliable_min_bit_errors"]),
            "stopping_receivers": list(eval_cfg["stopping_receivers"]),
        },
    }
    summary.update(artifacts)

    if PROPOSED_RECEIVER in enabled_receivers:
        summary["best_ber_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "ber")
        summary["best_bler_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "bler")
        summary["best_nmse_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "nmse")

    if classical_receivers:
        summary["best_ber_classical"] = _best_classical_row(classical_df, "ber", reliable_only=False)
        summary["best_bler_classical"] = _best_classical_row(classical_df, "bler", reliable_only=False)
        summary["best_nmse_classical"] = _best_classical_row(classical_df, "nmse", reliable_only=False)
        summary["best_ber_classical_reliable_only"] = _best_classical_row(classical_df, "ber", reliable_only=True)
        summary["best_bler_classical_reliable_only"] = _best_classical_row(classical_df, "bler", reliable_only=True)

    per_ebno_best_classical: list[dict[str, Any]] = []
    if classical_receivers and PROPOSED_RECEIVER in enabled_receivers:
        for ebno_db in sorted(df["ebno_db"].unique().tolist()):
            row_summary: dict[str, Any] = {"ebno_db": float(ebno_db)}
            classical_slice = classical_df[classical_df["ebno_db"] == ebno_db]
            proposed_slice = df[(df["receiver"] == PROPOSED_RECEIVER) & (df["ebno_db"] == ebno_db)]
            if proposed_slice.empty:
                continue
            proposed_row = proposed_slice.iloc[0]
            for metric in ["ber", "bler", "nmse"]:
                reliable_only = metric in {"ber", "bler"}
                best_classical = _best_classical_row(classical_slice, metric, reliable_only=reliable_only)
                if best_classical is None:
                    continue
                row_summary[f"best_{metric}_classical_receiver"] = best_classical["receiver"]
                row_summary[f"best_{metric}_classical"] = best_classical["value"]
                if pd.notna(proposed_row[metric]):
                    upair_value = float(proposed_row[metric])
                    row_summary[f"{metric}_upair5g"] = upair_value
                    row_summary[f"{metric}_gap_upair_minus_best_classical"] = upair_value - float(best_classical["value"])
                    if metric in {"ber", "bler"}:
                        row_summary[f"upair_{metric}_reliable"] = bool(proposed_row.get(f"reliable_{metric}", False))
            per_ebno_best_classical.append(row_summary)
    summary["per_ebno_best_classical"] = per_ebno_best_classical
    return summary


def _bool_cfg_list(cfg: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = get_cfg(cfg, key, default)
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _init_counter() -> dict[str, float | int]:
    return {
        "bit_errors": 0,
        "num_bits": 0,
        "block_errors": 0,
        "num_blocks": 0,
        "nmse_sum": 0.0,
        "num_nmse_batches": 0,
        "num_batches_run": 0,
    }


def _counter_progress_snapshot(counter: dict[str, float | int]) -> dict[str, float | int | None]:
    bit_errors = int(counter["bit_errors"])
    num_bits = int(counter["num_bits"])
    frame_errors = int(counter["block_errors"])
    num_frames = int(counter["num_blocks"])
    return {
        "frame_errors": frame_errors,
        "num_frames": num_frames,
        "bler": float(frame_errors / num_frames) if num_frames > 0 else None,
        "bit_errors": bit_errors,
        "num_bits": num_bits,
        "ber": float(bit_errors / num_bits) if num_bits > 0 else None,
        "num_batches_run": int(counter["num_batches_run"]),
    }


def _receiver_progress_snapshot(
    agg: dict[str, dict[str, float | int]],
    receiver_order: list[str],
) -> dict[str, dict[str, float | int | None]]:
    return {
        receiver_name: _counter_progress_snapshot(agg[receiver_name])
        for receiver_name in receiver_order
        if receiver_name in agg
    }


def _format_frame_error_progress(snapshot: dict[str, dict[str, float | int | None]]) -> str:
    parts = []
    for receiver_name, metrics in snapshot.items():
        parts.append(f"{receiver_name}:{int(metrics['frame_errors'])}/{int(metrics['num_frames'])}")
    return " ".join(parts)


def _update_error_counters(counter: dict[str, float | int], bits: tf.Tensor | None, b_hat: tf.Tensor | None, crc: tf.Tensor | None) -> None:
    if bits is not None and b_hat is not None:
        num_bits = int(tf.size(bits).numpy())
        ber_value = float(compute_ber(bits, b_hat).numpy())
        bit_errors = int(np.rint(ber_value * num_bits))
        counter["num_bits"] = int(counter["num_bits"]) + num_bits
        counter["bit_errors"] = int(counter["bit_errors"]) + bit_errors

    if crc is not None:
        num_blocks = int(tf.size(crc).numpy())
        bler_value = float(compute_bler_from_crc(crc).numpy())
        block_errors = int(np.rint(bler_value * num_blocks))
        counter["num_blocks"] = int(counter["num_blocks"]) + num_blocks
        counter["block_errors"] = int(counter["block_errors"]) + block_errors



def _receiver_error_counts_tensors(
    bits: tf.Tensor | None,
    b_hat: tf.Tensor | None,
    crc: tf.Tensor | None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return exact scalar error counts without materializing decoded bits in Python."""
    zero = tf.constant(0, tf.int64)
    bit_errors = zero
    num_bits = zero
    block_errors = zero
    num_blocks = zero

    if bits is not None and b_hat is not None:
        b_true_flat = flatten_bits(bits)
        b_hat_flat = flatten_bits(b_hat)
        n = tf.minimum(tf.size(b_true_flat, out_type=tf.int64), tf.size(b_hat_flat, out_type=tf.int64))
        b_true_flat = b_true_flat[:n]
        b_hat_flat = b_hat_flat[:n]
        bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(b_true_flat, b_hat_flat), tf.int64))
        num_bits = n

    if crc is not None:
        crc_tensor = tf.convert_to_tensor(crc)
        if crc_tensor.dtype != tf.bool:
            crc_bool = tf.cast(crc_tensor > 0, tf.bool)
        else:
            crc_bool = crc_tensor
        block_errors = tf.reduce_sum(tf.cast(tf.logical_not(crc_bool), tf.int64))
        num_blocks = tf.size(crc_bool, out_type=tf.int64)

    return bit_errors, num_bits, block_errors, num_blocks


def _add_error_counts(counter: dict[str, float | int], counts: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> None:
    bit_errors, num_bits, block_errors, num_blocks = counts
    counter["bit_errors"] = int(counter["bit_errors"]) + int(bit_errors.numpy())
    counter["num_bits"] = int(counter["num_bits"]) + int(num_bits.numpy())
    counter["block_errors"] = int(counter["block_errors"]) + int(block_errors.numpy())
    counter["num_blocks"] = int(counter["num_blocks"]) + int(num_blocks.numpy())


def _tensor_spec_like(value: tf.Tensor, *, flexible_batch_dim: bool = True) -> tf.TensorSpec:
    tensor = tf.convert_to_tensor(value)
    shape = tensor.shape.as_list()
    if flexible_batch_dim and shape:
        shape[0] = None
    return tf.TensorSpec(shape=shape, dtype=tensor.dtype)


def _compile_tf_function(fn: Any, input_signature: list[tf.TensorSpec], *, jit_compile: bool) -> Any:
    if jit_compile:
        try:
            return tf.function(fn, input_signature=input_signature, reduce_retracing=True, jit_compile=True)
        except TypeError:
            pass
    return tf.function(fn, input_signature=input_signature, reduce_retracing=True)


def _logical_microbatch_sizes(batch_size: int, microbatch_size: int) -> list[int]:
    batch_size = int(batch_size)
    microbatch_size = max(1, min(int(microbatch_size), batch_size))
    return [min(microbatch_size, batch_size - start) for start in range(0, batch_size, microbatch_size)]


class _ReceiverCallRunner:
    """Resolved receiver invocation and scalar-count wrapper for memory-stable eval.

    The old eval path called `call_receiver` for every microbatch. That helper tries
    several Keras/Sionna calling conventions until one works, so an invalid graph path
    can be built or partially allocated repeatedly. This runner resolves the valid
    convention once on a tiny batch and, when enabled, wraps the receiver plus BER/BLER
    counting inside one tf.function. Only four scalar counts cross back to Python.
    """

    _NO_H_MODES = ("list_y_no", "tuple_y_no", "positional_y_no")
    _WITH_H_MODES = (
        "positional_y_no_h",
        "list_y_no_h",
        "tuple_y_no_h",
        "positional_y_h_no",
        "list_y_h_no",
        "tuple_y_h_no",
    )

    def __init__(
        self,
        name: str,
        receiver: Any,
        sample_batch: dict[str, tf.Tensor],
        *,
        uses_h: bool,
        compile_error_counts: bool,
        jit_compile: bool,
    ) -> None:
        self.name = str(name)
        self.receiver = receiver
        self.uses_h = bool(uses_h)
        self.mode = self._resolve_mode(sample_batch)
        self._compiled_counter: Any | None = None
        self._compiled_enabled = False

        if compile_error_counts and sample_batch.get("b") is not None:
            self._compiled_counter = self._build_compiled_counter(sample_batch, jit_compile=jit_compile)
            self._compiled_enabled = self._compiled_counter is not None

        print(
            f"[EVAL] receiver_call name={self.name} mode={self.mode} "
            f"compiled_error_counts={self._compiled_enabled}"
        )

    def _call_mode(self, mode: str, y: tf.Tensor, no: tf.Tensor, h: tf.Tensor | None = None) -> Any:
        if mode == "list_y_no":
            return self.receiver([y, no])
        if mode == "tuple_y_no":
            return self.receiver((y, no))
        if mode == "positional_y_no":
            return self.receiver(y, no)
        if h is None:
            raise ValueError(f"Receiver {self.name!r} mode {mode!r} requires h.")
        if mode == "positional_y_no_h":
            return self.receiver(y, no, h)
        if mode == "list_y_no_h":
            return self.receiver([y, no, h])
        if mode == "tuple_y_no_h":
            return self.receiver((y, no, h))
        if mode == "positional_y_h_no":
            return self.receiver(y, h, no)
        if mode == "list_y_h_no":
            return self.receiver([y, h, no])
        if mode == "tuple_y_h_no":
            return self.receiver((y, h, no))
        raise ValueError(f"Unknown receiver call mode {mode!r} for {self.name!r}.")

    def _raw_call(self, y: tf.Tensor, no: tf.Tensor, h: tf.Tensor | None = None) -> Any:
        return self._call_mode(self.mode, y, no, h)

    def _resolve_mode(self, sample_batch: dict[str, tf.Tensor]) -> str:
        modes = self._WITH_H_MODES if self.uses_h else self._NO_H_MODES
        last_err: Exception | None = None
        for mode in modes:
            try:
                out = self._call_mode(
                    mode,
                    sample_batch["y"],
                    sample_batch["no"],
                    sample_batch.get("h") if self.uses_h else None,
                )
                b_hat, crc = infer_receiver_output(out)
                # Force actual execution on the tiny warm-up batch so an invalid calling
                # convention does not silently survive as a deferred graph failure.
                _ = int(tf.size(b_hat).numpy())
                if crc is not None:
                    _ = int(tf.size(crc).numpy())
                del out, b_hat, crc
                return mode
            except (tf.errors.ResourceExhaustedError, MemoryError):
                raise
            except Exception as exc:  # pragma: no cover - runtime compatibility helper
                last_err = exc
        raise RuntimeError(f"All receiver calling conventions failed for {self.name!r}.") from last_err

    def _build_compiled_counter(self, sample_batch: dict[str, tf.Tensor], *, jit_compile: bool) -> Any | None:
        bits_sig = _tensor_spec_like(sample_batch["b"])
        y_sig = _tensor_spec_like(sample_batch["y"])
        no_sig = _tensor_spec_like(sample_batch["no"])

        if self.uses_h:
            h_sig = _tensor_spec_like(sample_batch["h"])

            def counter_fn(bits: tf.Tensor, y: tf.Tensor, no: tf.Tensor, h: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                b_hat, crc = infer_receiver_output(self._raw_call(y, no, h))
                return _receiver_error_counts_tensors(bits, b_hat, crc)

            compiled = _compile_tf_function(counter_fn, [bits_sig, y_sig, no_sig, h_sig], jit_compile=jit_compile)
            warm_args = (sample_batch["b"], sample_batch["y"], sample_batch["no"], sample_batch["h"])
        else:

            def counter_fn(bits: tf.Tensor, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                b_hat, crc = infer_receiver_output(self._raw_call(y, no))
                return _receiver_error_counts_tensors(bits, b_hat, crc)

            compiled = _compile_tf_function(counter_fn, [bits_sig, y_sig, no_sig], jit_compile=jit_compile)
            warm_args = (sample_batch["b"], sample_batch["y"], sample_batch["no"])

        try:
            counts = compiled(*warm_args)
            # Materialize scalar counts during warm-up to catch graph-building issues now.
            _ = tuple(int(x.numpy()) for x in counts)
            del counts
            return compiled
        except (tf.errors.ResourceExhaustedError, MemoryError):
            raise
        except Exception as exc:  # pragma: no cover - depends on Sionna/Keras runtime
            print(
                f"[EVAL] receiver_call name={self.name} mode={self.mode} "
                f"compiled_error_counts=False reason={type(exc).__name__}: {exc}"
            )
            return None

    def update_counter(
        self,
        counter: dict[str, float | int],
        bits: tf.Tensor,
        y: tf.Tensor,
        no: tf.Tensor,
        h: tf.Tensor | None = None,
    ) -> None:
        if self._compiled_counter is not None:
            if self.uses_h:
                if h is None:
                    raise ValueError(f"Receiver {self.name!r} requires h for perfect-CSI evaluation.")
                counts = self._compiled_counter(bits, y, no, h)
            else:
                counts = self._compiled_counter(bits, y, no)
            _add_error_counts(counter, counts)
            del counts
            return

        b_hat, crc = infer_receiver_output(self._raw_call(y, no, h if self.uses_h else None))
        counts = _receiver_error_counts_tensors(bits, b_hat, crc)
        _add_error_counts(counter, counts)
        del b_hat, crc, counts


def _should_stop(
    agg: dict[str, dict[str, float | int]],
    stopping_receivers: list[str],
    batches_run: int,
    min_num_batches: int,
    max_num_batches: int,
    target_block_errors: int,
) -> bool:
    if batches_run < min_num_batches:
        return False
    if batches_run >= max_num_batches:
        return True
    if target_block_errors <= 0:
        return False
    for receiver_name in stopping_receivers:
        block_errors = int(agg[receiver_name]["block_errors"])
        if block_errors < target_block_errors:
            return False
    return True


def _target_floor_frames(target_bler_floor: float, confidence_factor: float) -> int:
    if target_bler_floor <= 0.0 or confidence_factor <= 0.0:
        return 0
    return int(math.ceil(confidence_factor / target_bler_floor))


def _receiver_stop_reason(
    counter: dict[str, float | int],
    receiver_name: str,
    *,
    batches_run: int,
    min_num_batches: int,
    max_num_batches: int,
    target_block_errors: int,
    stopping_receivers: set[str],
    floor_num_frames: int,
) -> str | None:
    if batches_run < min_num_batches:
        return None
    if batches_run >= max_num_batches:
        return "max_batches"

    block_errors = int(counter["block_errors"])
    num_blocks = int(counter["num_blocks"])
    if receiver_name in stopping_receivers and target_block_errors > 0 and block_errors >= target_block_errors:
        return f"target_block_errors_{target_block_errors}"
    if floor_num_frames > 0 and num_blocks >= floor_num_frames:
        if block_errors == 0:
            return "zero_error_floor"
        return "bler_floor_sample_budget"
    return None


def evaluate_model(cfg: dict[str, Any], checkpoint_path: str | None = None, num_users: int | None = None) -> dict[str, Any]:
    evaluation_start_s = time.perf_counter()
    training_seed = int(get_cfg(cfg, "system.training_seed", cfg["system"]["seed"]))
    evaluation_seed = int(get_cfg(cfg, "system.evaluation_seed", cfg["system"]["seed"]))
    set_global_seed(evaluation_seed)
    if bool(get_cfg(cfg, "system.graph_mode", True)):
        tf.config.run_functions_eagerly(False)
    paths = ensure_output_tree(cfg)
    curves_path = paths["metrics"] / "curves.csv"
    eval_state_path = paths["metrics"] / "evaluation_state.json"
    save_yaml(cfg, paths["artifacts"] / "resolved_config.yaml")

    eval_num_users = int(num_users if num_users is not None else get_cfg(cfg, "multiuser.fixed_num_users", max_num_users(cfg) if multiuser_enabled(cfg) else 1))
    tx, _ = build_pusch_transmitter(cfg, num_users=eval_num_users)
    channel = build_channel(cfg, tx)
    eval_batch_size = int(cfg["system"]["batch_size_eval"])
    receiver_microbatch_size = int(get_cfg(cfg, "evaluation.receiver_microbatch_size", eval_batch_size))
    if receiver_microbatch_size <= 0:
        receiver_microbatch_size = eval_batch_size
    receiver_microbatch_size = max(1, min(receiver_microbatch_size, eval_batch_size))
    if receiver_microbatch_size < eval_batch_size:
        print(
            "[EVAL] receiver microbatching enabled: "
            f"batch_size_eval={eval_batch_size} receiver_microbatch_size={receiver_microbatch_size}"
        )

    resource_grid = get_resource_grid(tx)
    pilot_mask = extract_true_dmrs_mask_per_stream(tx, resource_grid)
    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=resource_grid, cfg=cfg, pilot_mask=pilot_mask)

    warmup_batch = _make_eval_batch(
        tx=tx,
        channel=channel,
        cfg=cfg,
        batch_size=receiver_microbatch_size,
        ebno_db=float(get_cfg(cfg, "system.ebno_db_eval", [10])[0]),
    )
    estimator.estimate_with_ls(warmup_batch["y"], warmup_batch["no"], training=False)
    del warmup_batch
    _release_eval_memory()

    if checkpoint_path is not None:
        estimator.load_weights(str(checkpoint_path))
        print(f"[EVAL] loaded UPAIR checkpoint: {checkpoint_path}")

    enabled_receivers = enabled_receivers_from_cfg(cfg)
    classical_receivers, classical_estimators, baseline_artifacts = build_classical_baseline_suite(
        cfg=cfg,
        tx=tx,
        channel=channel,
        paths=paths,
    )

    proposed_rx = None
    if wants_receiver(cfg, PROPOSED_RECEIVER):
        proposed_rx = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)

    perfect_rx = None
    if wants_receiver(cfg, PERFECT_RECEIVER):
        perfect_rx = build_receiver(tx, cfg, channel_estimator=None, perfect_csi=True)

    ebno_grid = [float(x) for x in get_cfg(cfg, "system.ebno_db_eval", [0, 4, 8, 12])]

    max_num_batches = int(get_cfg(cfg, "evaluation.max_num_batches_per_point", get_cfg(cfg, "evaluation.num_batches_per_point", 256)))
    min_num_batches = int(get_cfg(cfg, "evaluation.min_num_batches_per_point", min(64, max_num_batches)))
    target_block_errors = int(get_cfg(cfg, "evaluation.target_block_errors_per_receiver", 0))
    reliable_min_block_errors = int(get_cfg(cfg, "evaluation.reliable_min_block_errors", 1))
    reliable_min_bit_errors = int(get_cfg(cfg, "evaluation.reliable_min_bit_errors", 1))
    stopping_receivers = _bool_cfg_list(cfg, "evaluation.stopping_receivers", enabled_receivers)
    progress_every_batches = int(get_cfg(cfg, "evaluation.progress_every_batches", 0))
    nmse_receivers = set(_bool_cfg_list(cfg, "evaluation.nmse_receivers", enabled_receivers))
    memory_cleanup_every_batches = int(get_cfg(cfg, "evaluation.memory_cleanup_every_batches", 1))
    memory_cleanup_every_microbatch = bool(get_cfg(cfg, "evaluation.memory_cleanup_every_microbatch", False))
    stream_eval_microbatches = bool(get_cfg(cfg, "evaluation.stream_eval_microbatches", True))
    compiled_receiver_error_counts = bool(get_cfg(cfg, "evaluation.compiled_receiver_error_counts", True))
    receiver_call_jit_compile = bool(get_cfg(cfg, "evaluation.receiver_call_jit_compile", False))
    log_latency = bool(get_cfg(cfg, "evaluation.log_latency", True))
    log_gpu_memory = bool(get_cfg(cfg, "evaluation.log_gpu_memory", True))
    per_receiver_stopping = bool(get_cfg(cfg, "evaluation.per_receiver_stopping", False))
    target_bler_floor = float(get_cfg(cfg, "evaluation.target_bler_floor", 0.0))
    floor_confidence_factor = float(get_cfg(cfg, "evaluation.floor_confidence_factor", 3.0))
    floor_num_frames = _target_floor_frames(target_bler_floor, floor_confidence_factor) if per_receiver_stopping else 0
    skipped_nmse_receivers = [name for name in enabled_receivers if name not in nmse_receivers]
    if skipped_nmse_receivers:
        print(
            "[EVAL] skipping explicit NMSE channel-estimator calls for: "
            + ", ".join(skipped_nmse_receivers)
        )
    print(
        "[EVAL] instrumentation "
        f"log_latency={log_latency} log_gpu_memory={log_gpu_memory} "
        f"memory_cleanup_every_batches={memory_cleanup_every_batches} "
        f"memory_cleanup_every_microbatch={memory_cleanup_every_microbatch} "
        f"stream_eval_microbatches={stream_eval_microbatches} "
        f"compiled_receiver_error_counts={compiled_receiver_error_counts} "
        f"receiver_call_jit_compile={receiver_call_jit_compile} "
        f"per_receiver_stopping={per_receiver_stopping} "
        f"target_bler_floor={target_bler_floor:g} "
        f"floor_num_frames={floor_num_frames}"
        + (_gpu_memory_message() if log_gpu_memory else "")
    )

    logical_microbatch_sizes = _logical_microbatch_sizes(eval_batch_size, receiver_microbatch_size)
    runner_warmup_batch = _make_eval_batch(
        tx=tx,
        channel=channel,
        cfg=cfg,
        batch_size=1,
        ebno_db=float(ebno_grid[0]),
    )
    classical_runners: dict[str, _ReceiverCallRunner] = {}
    for receiver_name, receiver in classical_receivers.items():
        if receiver_name in enabled_receivers:
            classical_runners[receiver_name] = _ReceiverCallRunner(
                receiver_name,
                receiver,
                runner_warmup_batch,
                uses_h=False,
                compile_error_counts=compiled_receiver_error_counts,
                jit_compile=receiver_call_jit_compile,
            )
    proposed_runner: _ReceiverCallRunner | None = None
    if proposed_rx is not None:
        proposed_runner = _ReceiverCallRunner(
            PROPOSED_RECEIVER,
            proposed_rx,
            runner_warmup_batch,
            uses_h=False,
            compile_error_counts=compiled_receiver_error_counts,
            jit_compile=receiver_call_jit_compile,
        )
    perfect_runner: _ReceiverCallRunner | None = None
    if perfect_rx is not None:
        perfect_runner = _ReceiverCallRunner(
            PERFECT_RECEIVER,
            perfect_rx,
            runner_warmup_batch,
            uses_h=True,
            compile_error_counts=compiled_receiver_error_counts,
            jit_compile=receiver_call_jit_compile,
        )
    del runner_warmup_batch
    _release_eval_memory()

    rows: list[dict[str, Any]] = []
    completed_ebno: set[float] = set()
    resume_eval = bool(get_cfg(cfg, "evaluation.resume", True)) and not bool(get_cfg(cfg, "evaluation.force", False))
    if resume_eval and curves_path.exists():
        try:
            existing = pd.read_csv(curves_path)
            if "num_users" in existing.columns:
                existing = existing[existing["num_users"] == eval_num_users].copy()
            required_receivers = set(enabled_receivers)
            for ebno_value, group in existing.groupby("ebno_db"):
                if required_receivers.issubset(set(group["receiver"].astype(str))):
                    completed_ebno.add(float(ebno_value))
            rows = existing.to_dict("records")
            if completed_ebno:
                done = ", ".join(f"{x:g}" for x in sorted(completed_ebno))
                print(f"[EVAL] resuming num_users={eval_num_users}; completed Eb/N0 points: {done}")
        except Exception as exc:
                print(f"[EVAL] ignoring unreadable partial curves {curves_path}: {exc!r}")

    example_saved = False
    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[EVAL] received signal {signum}; will save completed Eb/N0 points and stop after current batch.")

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_stop)
        except Exception:
            pass

    eval_cfg = {
        "min_num_batches_per_point": min_num_batches,
        "max_num_batches_per_point": max_num_batches,
        "target_block_errors_per_receiver": target_block_errors,
        "per_receiver_stopping": per_receiver_stopping,
        "target_bler_floor": target_bler_floor,
        "floor_confidence_factor": floor_confidence_factor,
        "floor_num_frames": floor_num_frames,
        "reliable_min_block_errors": reliable_min_block_errors,
        "reliable_min_bit_errors": reliable_min_bit_errors,
        "stopping_receivers": stopping_receivers,
        "progress_every_batches": progress_every_batches,
        "nmse_receivers": [name for name in enabled_receivers if name in nmse_receivers],
        "memory_cleanup_every_batches": memory_cleanup_every_batches,
        "memory_cleanup_every_microbatch": memory_cleanup_every_microbatch,
        "stream_eval_microbatches": stream_eval_microbatches,
        "compiled_receiver_error_counts": compiled_receiver_error_counts,
        "receiver_call_jit_compile": receiver_call_jit_compile,
        "log_latency": log_latency,
        "log_gpu_memory": log_gpu_memory,
    }

    def _save_eval_state(
        *,
        complete: bool,
        reason: str,
        completed: set[float],
        current_ebno_db: float | None = None,
        partial_batches_run: int | None = None,
        current_receiver_metrics: dict[str, dict[str, float | int | None]] | None = None,
        active_receivers: set[str] | None = None,
        receiver_stop_reasons: dict[str, str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "num_users": int(eval_num_users),
            "completed_ebno_db": sorted(float(x) for x in completed),
            "curves_csv": str(curves_path),
            "evaluation_complete": bool(complete),
            "save_reason": reason,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "evaluation_parameters": {
                "batch_size_eval": int(eval_batch_size),
                "receiver_microbatch_size": int(receiver_microbatch_size),
                "seed": int(evaluation_seed),
                "training_seed": int(training_seed),
                "evaluation_seed": int(evaluation_seed),
                **eval_cfg,
            },
        }
        if current_ebno_db is not None:
            payload["current_ebno_db"] = float(current_ebno_db)
        if partial_batches_run is not None:
            payload["partial_batches_run"] = int(partial_batches_run)
        if current_receiver_metrics is not None:
            payload["current_receiver_metrics"] = current_receiver_metrics
            payload["current_frame_errors"] = {
                receiver_name: int(metrics["frame_errors"])
                for receiver_name, metrics in current_receiver_metrics.items()
            }
            payload["current_num_frames"] = {
                receiver_name: int(metrics["num_frames"])
                for receiver_name, metrics in current_receiver_metrics.items()
            }
        if active_receivers is not None:
            payload["active_receivers"] = sorted(str(x) for x in active_receivers)
        if receiver_stop_reasons is not None:
            payload["receiver_stop_reasons"] = dict(receiver_stop_reasons)
        if complete:
            payload["summary_path"] = str(paths["metrics"] / "evaluation_summary.json")
        save_json(payload, eval_state_path)

    try:
        for ebno_db in ebno_grid:
            if float(ebno_db) in completed_ebno:
                print(f"[EVAL] reusing completed Eb/N0={ebno_db:g} dB for num_users={eval_num_users}")
                continue

            agg: dict[str, dict[str, float | int]] = {
                receiver_name: _init_counter()
                for receiver_name in enabled_receivers
            }
            point_start_s = time.perf_counter()
            receiver_elapsed_s = {receiver_name: 0.0 for receiver_name in enabled_receivers}
            data_elapsed_s = 0.0
            point_interrupted = False
            batches_completed = 0
            active_receivers = set(enabled_receivers)
            receiver_stop_reasons: dict[str, str] = {}

            for batch_idx in range(max_num_batches):
                if per_receiver_stopping and not active_receivers:
                    break
                collect_example = not example_saved and bool(get_cfg(cfg, "evaluation.save_example_batch", True))
                example_classical_h_parts: dict[str, list[tf.Tensor]] = {name: [] for name in classical_estimators}
                example_h_prop_parts: list[tf.Tensor] = []
                example_h_ls_parts: list[tf.Tensor] = []
                example_h_true_parts: list[tf.Tensor] = []
                example_y_parts: list[tf.Tensor] = []
                nmse_acc: dict[str, list[float]] = {receiver_name: [0.0, 0.0] for receiver_name in enabled_receivers}
                receivers_evaluated_this_batch: set[str] = set()

                full_batch: dict[str, tf.Tensor] | None = None
                microbatch_iter: Iterator[dict[str, tf.Tensor]] | None = None
                if not stream_eval_microbatches:
                    batch_make_start_s = time.perf_counter()
                    full_batch = _make_eval_batch(
                        tx=tx,
                        channel=channel,
                        cfg=cfg,
                        batch_size=eval_batch_size,
                        ebno_db=ebno_db,
                    )
                    data_elapsed_s += time.perf_counter() - batch_make_start_s
                    microbatch_iter = _iter_eval_microbatches(full_batch, receiver_microbatch_size)

                for current_microbatch_size in logical_microbatch_sizes:
                    if stream_eval_microbatches:
                        batch_make_start_s = time.perf_counter()
                        micro_batch = _make_eval_batch(
                            tx=tx,
                            channel=channel,
                            cfg=cfg,
                            batch_size=current_microbatch_size,
                            ebno_db=ebno_db,
                        )
                        data_elapsed_s += time.perf_counter() - batch_make_start_s
                    else:
                        if microbatch_iter is None:
                            raise RuntimeError("Internal error: missing eval microbatch iterator.")
                        micro_batch = next(microbatch_iter)

                    if collect_example:
                        example_h_true_parts.append(micro_batch["h"])
                        example_y_parts.append(micro_batch["y"])

                    for receiver_name, estimator_block in classical_estimators.items():
                        if receiver_name not in classical_runners:
                            continue
                        if per_receiver_stopping and receiver_name not in active_receivers:
                            continue
                        receiver_start_s = time.perf_counter()
                        compute_nmse = receiver_name in nmse_receivers
                        h_hat_base: tf.Tensor | None = None
                        if compute_nmse or collect_example:
                            h_hat_base, _ = _call_channel_estimator(estimator_block, micro_batch["y"], micro_batch["no"])
                            if compute_nmse:
                                num, den = _nmse_components(micro_batch["h"], h_hat_base)
                                nmse_acc[receiver_name][0] += num
                                nmse_acc[receiver_name][1] += den
                            if collect_example:
                                example_classical_h_parts.setdefault(receiver_name, []).append(h_hat_base)
                            else:
                                del h_hat_base
                        classical_runners[receiver_name].update_counter(
                            agg[receiver_name],
                            micro_batch["b"],
                            micro_batch["y"],
                            micro_batch["no"],
                        )
                        receivers_evaluated_this_batch.add(receiver_name)
                        receiver_elapsed_s[receiver_name] = receiver_elapsed_s.get(receiver_name, 0.0) + (
                            time.perf_counter() - receiver_start_s
                        )

                    if proposed_runner is not None and (not per_receiver_stopping or PROPOSED_RECEIVER in active_receivers):
                        receiver_start_s = time.perf_counter()
                        compute_nmse = PROPOSED_RECEIVER in nmse_receivers
                        h_hat_prop: tf.Tensor | None = None
                        h_ls: tf.Tensor | None = None
                        if compute_nmse or collect_example:
                            h_hat_prop, _, h_ls, _ = estimator.estimate_with_ls(micro_batch["y"], micro_batch["no"], training=False)
                            if compute_nmse:
                                num, den = _nmse_components(micro_batch["h"], h_hat_prop)
                                nmse_acc[PROPOSED_RECEIVER][0] += num
                                nmse_acc[PROPOSED_RECEIVER][1] += den
                            if collect_example:
                                example_h_prop_parts.append(h_hat_prop)
                                example_h_ls_parts.append(h_ls)
                            else:
                                del h_hat_prop, h_ls
                        proposed_runner.update_counter(
                            agg[PROPOSED_RECEIVER],
                            micro_batch["b"],
                            micro_batch["y"],
                            micro_batch["no"],
                        )
                        receivers_evaluated_this_batch.add(PROPOSED_RECEIVER)
                        receiver_elapsed_s[PROPOSED_RECEIVER] = receiver_elapsed_s.get(PROPOSED_RECEIVER, 0.0) + (
                            time.perf_counter() - receiver_start_s
                        )

                    if perfect_runner is not None and (not per_receiver_stopping or PERFECT_RECEIVER in active_receivers):
                        receiver_start_s = time.perf_counter()
                        perfect_runner.update_counter(
                            agg[PERFECT_RECEIVER],
                            micro_batch["b"],
                            micro_batch["y"],
                            micro_batch["no"],
                            h=micro_batch["h"],
                        )
                        receivers_evaluated_this_batch.add(PERFECT_RECEIVER)
                        receiver_elapsed_s[PERFECT_RECEIVER] = receiver_elapsed_s.get(PERFECT_RECEIVER, 0.0) + (
                            time.perf_counter() - receiver_start_s
                        )

                    del micro_batch
                    if memory_cleanup_every_microbatch:
                        _release_eval_memory()

                if full_batch is not None:
                    del full_batch

                for receiver_name in receivers_evaluated_this_batch:
                    if receiver_name == PERFECT_RECEIVER:
                        # Perfect CSI has zero channel-estimation error by definition; keep
                        # the historical output convention of reporting NMSE=0 for this row.
                        agg[receiver_name]["num_nmse_batches"] = int(agg[receiver_name]["num_nmse_batches"]) + 1
                    elif receiver_name in nmse_receivers:
                        nmse_num, nmse_den = nmse_acc[receiver_name]
                        agg[receiver_name]["nmse_sum"] = float(agg[receiver_name]["nmse_sum"]) + float(nmse_num / max(nmse_den, 1e-9))
                        agg[receiver_name]["num_nmse_batches"] = int(agg[receiver_name]["num_nmse_batches"]) + 1
                    agg[receiver_name]["num_batches_run"] = int(agg[receiver_name]["num_batches_run"]) + 1

                if collect_example:
                    example_classical_h_hats: dict[str, tf.Tensor] = {}
                    for receiver_name, h_parts in example_classical_h_parts.items():
                        concatenated = _safe_concat(h_parts)
                        if concatenated is not None:
                            example_classical_h_hats[receiver_name] = concatenated
                    example_h_hat_prop = _safe_concat(example_h_prop_parts)
                    example_h_ls = _safe_concat(example_h_ls_parts)
                    example_h_true = _safe_concat(example_h_true_parts)
                    example_y = _safe_concat(example_y_parts)
                    if example_h_true is not None and example_y is not None:
                        example_payload: dict[str, Any] = {
                            "h_true": np.asarray(example_h_true.numpy()),
                            "y": np.asarray(example_y.numpy()),
                            "ebno_db": np.asarray([ebno_db]),
                        }
                        if "baseline_ls_lmmse" in example_classical_h_hats:
                            example_payload["h_ls_linear"] = np.asarray(example_classical_h_hats["baseline_ls_lmmse"].numpy())
                        elif example_h_ls is not None:
                            example_payload["h_ls_linear"] = np.asarray(example_h_ls.numpy())
                        if "baseline_ls_timeavg_lmmse" in example_classical_h_hats:
                            example_payload["h_ls_timeavg"] = np.asarray(example_classical_h_hats["baseline_ls_timeavg_lmmse"].numpy())
                        if "baseline_ls_2dlmmse_lmmse" in example_classical_h_hats:
                            example_payload["h_ls_2dlmmse"] = np.asarray(example_classical_h_hats["baseline_ls_2dlmmse_lmmse"].numpy())
                        if "baseline_ddcpe_ls_lmmse" in example_classical_h_hats:
                            example_payload["h_ddcpe_ls"] = np.asarray(example_classical_h_hats["baseline_ddcpe_ls_lmmse"].numpy())
                        if example_h_hat_prop is not None:
                            example_payload["h_prop"] = np.asarray(example_h_hat_prop.numpy())
                        np.savez_compressed(paths["artifacts"] / "channel_example.npz", **example_payload)
                        del example_payload
                    example_saved = True

                batches_completed = batch_idx + 1
                del example_classical_h_parts, example_h_prop_parts, example_h_ls_parts, example_h_true_parts, example_y_parts, nmse_acc
                if memory_cleanup_every_batches > 0 and batches_completed % memory_cleanup_every_batches == 0:
                    _release_eval_memory()

                if per_receiver_stopping:
                    for receiver_name in list(active_receivers):
                        stop_reason = _receiver_stop_reason(
                            agg[receiver_name],
                            receiver_name,
                            batches_run=batches_completed,
                            min_num_batches=min_num_batches,
                            max_num_batches=max_num_batches,
                            target_block_errors=target_block_errors,
                            stopping_receivers=set(stopping_receivers),
                            floor_num_frames=floor_num_frames,
                        )
                        if stop_reason is not None:
                            active_receivers.remove(receiver_name)
                            receiver_stop_reasons[receiver_name] = stop_reason
                            counter = agg[receiver_name]
                            num_blocks_done = int(counter["num_blocks"])
                            upper = (
                                floor_confidence_factor / num_blocks_done
                                if num_blocks_done > 0 and int(counter["block_errors"]) == 0
                                else float("nan")
                            )
                            print(
                                f"[EVAL] receiver_done receiver={receiver_name} "
                                f"Eb/N0={ebno_db:g} dB reason={stop_reason} "
                                f"batches={int(counter['num_batches_run'])} "
                                f"frame_err={int(counter['block_errors'])}/{num_blocks_done} "
                                f"zero_error_bler_upper={upper:.3e}"
                                + (_gpu_memory_message() if log_gpu_memory else "")
                            )

                if progress_every_batches > 0 and batches_completed % progress_every_batches == 0:
                    progress_snapshot = _receiver_progress_snapshot(agg, enabled_receivers)
                    _save_eval_state(
                        complete=False,
                        reason="progress",
                        completed=completed_ebno,
                        current_ebno_db=float(ebno_db),
                        partial_batches_run=batches_completed,
                        current_receiver_metrics=progress_snapshot,
                        active_receivers=active_receivers,
                        receiver_stop_reasons=receiver_stop_reasons,
                    )
                    print(
                        f"[EVAL] progress num_users={eval_num_users} "
                        f"Eb/N0={ebno_db:g} dB batches={batches_completed}/{max_num_batches} "
                        f"frame_err={_format_frame_error_progress(progress_snapshot)}"
                        + (
                            f" elapsed={time.perf_counter() - point_start_s:.1f}s "
                            f"avg_batch={(time.perf_counter() - point_start_s) / max(batches_completed, 1):.3f}s"
                            if log_latency
                            else ""
                        )
                        + (_gpu_memory_message() if log_gpu_memory else "")
                    )

                if stop_requested:
                    point_interrupted = True
                    break

                if per_receiver_stopping:
                    if not active_receivers:
                        break
                else:
                    if _should_stop(
                        agg=agg,
                        stopping_receivers=[r for r in stopping_receivers if r in agg],
                        batches_run=batch_idx + 1,
                        min_num_batches=min_num_batches,
                        max_num_batches=max_num_batches,
                        target_block_errors=target_block_errors,
                    ):
                        break

            if point_interrupted:
                pd.DataFrame(rows).to_csv(curves_path, index=False)
                _save_eval_state(
                    complete=False,
                    reason="signal",
                    completed=completed_ebno,
                    current_ebno_db=float(ebno_db),
                    partial_batches_run=batches_completed,
                    current_receiver_metrics=_receiver_progress_snapshot(agg, enabled_receivers),
                    active_receivers=active_receivers,
                    receiver_stop_reasons=receiver_stop_reasons,
                )
                print("[EVAL] stopped before completing current Eb/N0; resubmit to resume from completed points.")
                return {
                    "output_dir": str(paths["root"]),
                    "curves_path": str(curves_path),
                    "summary_path": str(paths["metrics"] / "evaluation_summary.json"),
                    "evaluation_complete": False,
                    "completed_ebno_db": sorted(float(x) for x in completed_ebno),
                    "training_seed": int(training_seed),
                    "evaluation_seed": int(evaluation_seed),
                }

            for receiver_name in enabled_receivers:
                counter = agg[receiver_name]
                point_elapsed_s = time.perf_counter() - point_start_s
                receiver_time_s = float(receiver_elapsed_s.get(receiver_name, 0.0))
                num_bits = int(counter["num_bits"])
                num_blocks = int(counter["num_blocks"])
                bit_errors = int(counter["bit_errors"])
                block_errors = int(counter["block_errors"])
                num_nmse_batches = int(counter["num_nmse_batches"])
                bler_upper_bound = (
                    float(floor_confidence_factor / num_blocks)
                    if num_blocks > 0 and block_errors == 0 and floor_confidence_factor > 0.0
                    else np.nan
                )
                ber_upper_bound = (
                    float(floor_confidence_factor / num_bits)
                    if num_bits > 0 and bit_errors == 0 and floor_confidence_factor > 0.0
                    else np.nan
                )

                row = {
                    "receiver": receiver_name,
                    "num_users": eval_num_users,
                    "ebno_db": ebno_db,
                    "ber": float(bit_errors / num_bits) if num_bits > 0 else np.nan,
                    "bler": float(block_errors / num_blocks) if num_blocks > 0 else np.nan,
                    "nmse": float(counter["nmse_sum"] / num_nmse_batches) if num_nmse_batches > 0 else np.nan,
                    "bit_errors": bit_errors,
                    "num_bits": num_bits,
                    "block_errors": block_errors,
                    "num_blocks": num_blocks,
                    "num_batches_run": int(counter["num_batches_run"]),
                    "mc_stop_reason": receiver_stop_reasons.get(receiver_name, "global_stop"),
                    "target_bler_floor": float(target_bler_floor),
                    "bler_zero_error_upper_bound": bler_upper_bound,
                    "ber_zero_error_upper_bound": ber_upper_bound,
                    "point_elapsed_s": float(point_elapsed_s),
                    "data_elapsed_s": float(data_elapsed_s),
                    "receiver_elapsed_s": float(receiver_time_s),
                    "receiver_ms_per_batch": float(1000.0 * receiver_time_s / max(int(counter["num_batches_run"]), 1)),
                    "receiver_ms_per_frame": float(1000.0 * receiver_time_s / max(num_blocks, 1)),
                    "reliable_ber": bool(bit_errors >= reliable_min_bit_errors),
                    "reliable_bler": bool(block_errors >= reliable_min_block_errors),
                    **_gpu_memory_stats(),
                }
                rows.append(row)
                print(
                    f"[EVAL] receiver={receiver_name:>24s} "
                    f"Eb/N0={ebno_db:>4.1f} dB "
                    f"BER={row['ber']:.5e} "
                    f"BLER={row['bler']:.5e} "
                    f"NMSE={row['nmse']:.5e} "
                    f"bit_err={bit_errors:>6d}/{num_bits:<8d} "
                    f"blk_err={block_errors:>5d}/{num_blocks:<6d} "
                    f"batches={int(counter['num_batches_run']):>4d}"
                    f" stop={row['mc_stop_reason']}"
                    + (
                        f" bler_ub={row['bler_zero_error_upper_bound']:.3e}"
                        if np.isfinite(row["bler_zero_error_upper_bound"])
                        else ""
                    )
                    + (
                        f" rx_time={receiver_time_s:.2f}s "
                        f"rx_ms_batch={row['receiver_ms_per_batch']:.1f} "
                        f"rx_ms_frame={row['receiver_ms_per_frame']:.3f} "
                        f"point_time={point_elapsed_s:.2f}s"
                        if log_latency
                        else ""
                    )
                    + (_gpu_memory_message() if log_gpu_memory else "")
                )

            completed_ebno.add(float(ebno_db))
            pd.DataFrame(rows).to_csv(curves_path, index=False)
            _save_eval_state(
                complete=False,
                reason="periodic",
                completed=completed_ebno,
                current_ebno_db=float(ebno_db),
                partial_batches_run=batches_completed,
                current_receiver_metrics=_receiver_progress_snapshot(agg, enabled_receivers),
                active_receivers=active_receivers,
                receiver_stop_reasons=receiver_stop_reasons,
            )
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df.to_csv(curves_path, index=False)
    summary = _build_summary(
        df=df,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        enabled_receivers=enabled_receivers,
        artifacts=baseline_artifacts,
        eval_cfg=eval_cfg,
    )
    summary["curves_csv"] = str(curves_path)
    summary["batch_size_eval"] = int(eval_batch_size)
    summary["receiver_microbatch_size"] = int(receiver_microbatch_size)
    summary["evaluation_elapsed_s"] = float(time.perf_counter() - evaluation_start_s)
    summary.update(_gpu_memory_stats())
    summary["seed"] = int(evaluation_seed)
    summary["training_seed"] = int(training_seed)
    summary["evaluation_seed"] = int(evaluation_seed)
    save_json(summary, paths["metrics"] / "evaluation_summary.json")
    _save_eval_state(complete=True, reason="final", completed=set(float(x) for x in ebno_grid))

    return {
        "output_dir": str(paths["root"]),
        "curves_path": str(curves_path),
        "summary_path": str(paths["metrics"] / "evaluation_summary.json"),
        "evaluation_complete": True,
        "completed_ebno_db": [float(x) for x in ebno_grid],
        "training_seed": int(training_seed),
        "evaluation_seed": int(evaluation_seed),
    }
