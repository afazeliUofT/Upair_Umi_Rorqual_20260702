from __future__ import annotations

"""Impairment handling for UPAIR experiments.

Default policy for the Narval multi-user extension:
  * The old explicit symbol-wise phase equation from the paper is disabled by
    default and kept only as a legacy/debug option.
  * A verified Sionna OFDM waveform-domain RF impairment path is provided for
    CFO and oscillator-like phase noise:

        PUSCH frequency grid -> Sionna OFDMModulator
        -> time-domain CFO/phase-noise phasor
        -> Sionna OFDMDemodulator -> existing channel/receiver pipeline

    This path was enabled only after the Narval probes confirmed the multi-user
    OFDMModulator/OFDMDemodulator tensor shapes. It intentionally does not
    replace the existing CDL OFDMChannel with TimeChannel because the probe did
    not verify a publication-safe conversion from TimeChannel taps to the exact
    per-RE channel tensor used by the current supervised NMSE loss and perfect-
    CSI receiver.
"""

import math
from typing import Any

import numpy as np
import tensorflow as tf

from .compat import safe_call_variants
from .config import get_cfg


class RFImpairmentNotImplementedError(NotImplementedError):
    """Raised when a requested RF backend is not implemented safely."""


# -----------------------------------------------------------------------------
# Legacy paper-equation impairment: disabled by default
# -----------------------------------------------------------------------------

def _sample_legacy_phase_profile(
    batch_size: int | tf.Tensor,
    num_symbols: tf.Tensor,
    impair_cfg: dict[str, Any],
    training: bool,
) -> tf.Tensor:
    """Sample the old simplified symbol-wise phase profile.

    This is the explicit equation used in the earlier paper draft. It is not
    used by default in the Narval package. Keep it only for ablation/debugging.
    """
    batch_size = tf.cast(batch_size, tf.int32)
    num_symbols = tf.cast(num_symbols, tf.int32)
    t = tf.cast(tf.range(num_symbols), tf.float32)[tf.newaxis, :]  # [1, T]

    cpe_sigma = float(impair_cfg.get("cpe_sigma_rad", 0.0))
    rw_sigma = float(impair_cfg.get("rw_sigma_rad", 0.0))

    phi0 = tf.random.normal(tf.stack([batch_size, 1]), stddev=cpe_sigma)

    if training and "slope_range_rad" in impair_cfg:
        low, high = impair_cfg.get("slope_range_rad", [0.0, 0.0])
        slope = tf.random.uniform(tf.stack([batch_size, 1]), minval=float(low), maxval=float(high))
    else:
        slope = tf.fill(tf.stack([batch_size, 1]), float(impair_cfg.get("slope_rad", 0.0)))

    centered_t = t - tf.cast(num_symbols - 1, tf.float32) / 2.0
    linear = slope * centered_t / tf.maximum(tf.cast(num_symbols - 1, tf.float32), 1.0)

    rw_increments = tf.random.normal(tf.stack([batch_size, num_symbols]), stddev=rw_sigma)
    rw = tf.cumsum(rw_increments, axis=1)
    return phi0 + linear + rw  # [B, T]


def _apply_legacy_symbol_phase(
    y: tf.Tensor,
    h: tf.Tensor | None,
    cfg: dict[str, Any],
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor | None]:
    if "legacy_phase_impairments" in cfg:
        enabled = bool(get_cfg(cfg, "legacy_phase_impairments.enabled", False))
        impair_cfg = get_cfg(cfg, f"legacy_phase_impairments.{ 'train' if training else 'eval' }", {}) or {}
    else:
        enabled = bool(get_cfg(cfg, "impairments.enabled", False))
        impair_cfg = get_cfg(cfg, f"impairments.{ 'train' if training else 'eval' }", {}) or {}

    if not enabled:
        return y, h

    num_symbols = tf.shape(y)[-2]
    batch_size = tf.shape(y)[0]
    phase = _sample_legacy_phase_profile(batch_size, num_symbols, impair_cfg, training=training)

    phasor_y = tf.cast(tf.exp(tf.complex(tf.zeros_like(phase), phase)), y.dtype)
    phasor_y = tf.reshape(phasor_y, [tf.shape(y)[0], 1, 1, tf.shape(y)[-2], 1])
    y_out = y * phasor_y

    if h is None:
        return y_out, None

    phasor_h = tf.cast(tf.exp(tf.complex(tf.zeros_like(phase), phase)), h.dtype)
    phasor_h = tf.reshape(phasor_h, [tf.shape(h)[0], 1, 1, 1, 1, tf.shape(h)[-2], 1])
    h_out = h * phasor_h
    return y_out, h_out


# -----------------------------------------------------------------------------
# Verified Sionna OFDM waveform-domain RF impairment path
# -----------------------------------------------------------------------------

def _rf_impairments_enabled(cfg: dict[str, Any]) -> bool:
    return bool(get_cfg(cfg, "rf_impairments.enabled", False))


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "none": "clean",
        "off": "clean",
        "no_impairment": "clean",
        "cfo": "cfo_only",
        "phase": "phase_noise_only",
        "phase_noise": "phase_noise_only",
        "pn": "phase_noise_only",
        "both": "cfo_and_phase_noise",
        "impaired": "cfo_and_phase_noise",
        "cfo_phase": "cfo_and_phase_noise",
        "cfo_pn": "cfo_and_phase_noise",
    }
    return aliases.get(mode, mode)


def _sample_training_rf_mode(cfg: dict[str, Any]) -> str:
    mixture = get_cfg(cfg, "rf_impairments.train_mixture", {}) or {}
    enabled = bool(mixture.get("enabled", True))
    if not enabled:
        return _normalize_mode(str(get_cfg(cfg, "rf_impairments.train_mode", "cfo_and_phase_noise")))

    entries = [
        ("clean", float(mixture.get("clean_probability", 0.0))),
        ("cfo_only", float(mixture.get("cfo_only_probability", 0.0))),
        ("phase_noise_only", float(mixture.get("phase_noise_only_probability", 0.0))),
        ("cfo_and_phase_noise", float(mixture.get("cfo_and_phase_noise_probability", 0.0))),
    ]
    modes = [m for m, w in entries]
    weights = np.asarray([max(0.0, w) for _, w in entries], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        # Sensible default for robust training if probabilities were omitted.
        weights = np.asarray([0.30, 0.25, 0.25, 0.20], dtype=np.float64)
    weights = weights / weights.sum()
    return str(np.random.choice(modes, p=weights))


def _rf_mode(cfg: dict[str, Any], training: bool) -> str:
    if not _rf_impairments_enabled(cfg):
        return "clean"
    if training:
        return _sample_training_rf_mode(cfg)
    return _normalize_mode(str(get_cfg(cfg, "rf_impairments.eval.mode", "cfo_and_phase_noise")))


def _build_ofdm_modem(tx: Any, cfg: dict[str, Any]) -> tuple[Any, Any, float]:
    """Build Sionna OFDM modulator/demodulator for the current resource grid."""
    from .builders import get_resource_grid  # local import avoids circular imports

    try:
        from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator
    except Exception:  # pragma: no cover - compatibility with older Sionna
        from sionna.ofdm import OFDMModulator, OFDMDemodulator

    rg = get_resource_grid(tx)
    fft_size = int(getattr(rg, "fft_size"))
    cp_len = int(getattr(rg, "cyclic_prefix_length"))
    # ResourceGrid.subcarrier_spacing is in Hz in Sionna 1.2.1. Fall back to config.
    scs = getattr(rg, "subcarrier_spacing", None)
    if scs is None:
        scs_hz = float(get_cfg(cfg, "pusch.subcarrier_spacing_khz", 30.0)) * 1e3
    else:
        scs_hz = float(scs)
        if scs_hz < 1e3:
            scs_hz *= 1e3
    sample_rate_hz = float(fft_size) * float(scs_hz)
    precision = get_cfg(cfg, "system.precision", "single")

    try:
        mod = OFDMModulator(cyclic_prefix_length=cp_len, precision=precision)
    except Exception:
        mod = OFDMModulator(cyclic_prefix_length=cp_len)

    # l_min=0 is the verified roundtrip convention from the Narval probe for the
    # OFDM-only waveform augmentation path.
    try:
        demod = OFDMDemodulator(fft_size=fft_size, l_min=0, cyclic_prefix_length=cp_len, precision=precision)
    except Exception:
        demod = OFDMDemodulator(fft_size, 0, cp_len)

    return mod, demod, sample_rate_hz


def _cfo_values_hz(batch: tf.Tensor, users: tf.Tensor, cfg: dict[str, Any], training: bool) -> tf.Tensor:
    if training:
        distribution = str(get_cfg(cfg, "rf_impairments.cfo.distribution", "uniform")).lower()
        max_abs = float(get_cfg(cfg, "rf_impairments.cfo.train_max_abs_hz", 0.0))
        if distribution == "normal":
            std = float(get_cfg(cfg, "rf_impairments.cfo.train_std_hz", max_abs / 2.0 if max_abs > 0.0 else 0.0))
            values = tf.random.normal(tf.stack([batch, users, tf.constant(1, tf.int32), tf.constant(1, tf.int32)]), stddev=std, dtype=tf.float32)
            if max_abs > 0.0:
                values = tf.clip_by_value(values, -max_abs, max_abs)
            return values
        return tf.random.uniform(tf.stack([batch, users, tf.constant(1, tf.int32), tf.constant(1, tf.int32)]), minval=-max_abs, maxval=max_abs, dtype=tf.float32)

    eval_hz = float(get_cfg(cfg, "rf_impairments.cfo.eval_hz", 0.0))
    per_user = get_cfg(cfg, "rf_impairments.cfo.eval_hz_per_user", None)
    if per_user is not None:
        values = tf.convert_to_tensor([float(x) for x in per_user], dtype=tf.float32)
        values = values[:users]
        values = tf.reshape(values, [1, -1, 1, 1])
        values = tf.broadcast_to(values, tf.stack([batch, users, tf.constant(1, tf.int32), tf.constant(1, tf.int32)]))
        return values
    return tf.fill(tf.stack([batch, users, tf.constant(1, tf.int32), tf.constant(1, tf.int32)]), tf.cast(eval_hz, tf.float32))


def _phase_noise_params(cfg: dict[str, Any], training: bool) -> tuple[float, float]:
    prefix = "train" if training else "eval"
    rms = float(get_cfg(cfg, f"rf_impairments.phase_noise.{prefix}_rms_rad_per_sample", 0.0))
    # Optional initial random phase models residual oscillator phase at the
    # beginning of the waveform. It is not the old OFDM-symbol equation.
    init = float(get_cfg(cfg, f"rf_impairments.phase_noise.{prefix}_initial_phase_sigma_rad", 0.0))
    return rms, init


def _time_domain_rf_phasor(
    x_time: tf.Tensor,
    cfg: dict[str, Any],
    training: bool,
    mode: str,
    sample_rate_hz: float,
) -> tf.Tensor:
    x_time = tf.convert_to_tensor(x_time)
    if x_time.shape.rank != 4:
        raise ValueError(f"Expected time waveform rank 4 [B,U,S,N], got rank {x_time.shape.rank}.")

    b = tf.shape(x_time)[0]
    u = tf.shape(x_time)[1]
    n_samp = tf.shape(x_time)[-1]
    n = tf.cast(tf.range(n_samp), tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
    phase = tf.zeros(tf.stack([b, u, tf.constant(1, tf.int32), n_samp]), dtype=tf.float32)

    cfo_enabled = bool(get_cfg(cfg, "rf_impairments.cfo.enabled", False)) and mode in {"cfo_only", "cfo_and_phase_noise"}
    if cfo_enabled:
        cfo = _cfo_values_hz(b, u, cfg, training=training)
        phase = phase + 2.0 * math.pi * cfo * n / float(sample_rate_hz)

    pn_enabled = bool(get_cfg(cfg, "rf_impairments.phase_noise.enabled", False)) and mode in {"phase_noise_only", "cfo_and_phase_noise"}
    if pn_enabled:
        rms, init_sigma = _phase_noise_params(cfg, training=training)
        if init_sigma > 0.0:
            init_phase = tf.random.normal(tf.stack([b, u, tf.constant(1, tf.int32), tf.constant(1, tf.int32)]), stddev=init_sigma, dtype=tf.float32)
            phase = phase + init_phase
        if rms > 0.0:
            increments = tf.random.normal(tf.stack([b, u, tf.constant(1, tf.int32), n_samp]), stddev=rms, dtype=tf.float32)
            phase = phase + tf.cumsum(increments, axis=-1)

    return tf.exp(tf.complex(tf.zeros_like(phase), phase))


def apply_rf_impairments_to_transmit_grid_if_enabled(
    x: tf.Tensor,
    tx: Any,
    cfg: dict[str, Any],
    training: bool,
) -> tuple[tf.Tensor, dict[str, Any]]:
    """Apply verified waveform-domain RF impairments to the transmit grid.

    Returns the possibly modified frequency-domain grid and a metadata dict.
    In clean mode, the input tensor is returned unchanged.
    """
    mode = _rf_mode(cfg, training=training)
    meta: dict[str, Any] = {"rf_mode": mode, "rf_enabled": bool(_rf_impairments_enabled(cfg))}
    if mode == "clean" or not _rf_impairments_enabled(cfg):
        return x, meta

    backend = str(get_cfg(cfg, "rf_impairments.backend", "sionna_ofdm_waveform_prechannel")).lower()
    if backend not in {"sionna_ofdm_waveform_prechannel", "ofdm_waveform_prechannel", "sionna_time_domain_prechannel"}:
        raise RFImpairmentNotImplementedError(
            "Requested rf_impairments.backend={!r}. The implemented safe backend is "
            "'sionna_ofdm_waveform_prechannel'. A full TimeChannel replacement is "
            "not enabled until exact effective-channel labels are verified.".format(backend)
        )

    x = tf.convert_to_tensor(x)
    mod, demod, sample_rate_hz = _build_ofdm_modem(tx, cfg)
    x_time = safe_call_variants(mod, x)
    phasor = _time_domain_rf_phasor(x_time, cfg=cfg, training=training, mode=mode, sample_rate_hz=sample_rate_hz)
    x_time_imp = x_time * tf.cast(phasor, x_time.dtype)
    x_imp = safe_call_variants(demod, x_time_imp)
    x_imp = tf.cast(x_imp, x.dtype)

    if x.shape.rank is not None and x_imp.shape.rank is not None and x.shape.rank != x_imp.shape.rank:
        raise ValueError(f"RF OFDM roundtrip changed rank from {x.shape.rank} to {x_imp.shape.rank}.")
    if x.shape.rank is not None and x.shape.rank >= 1:
        # Dynamic shape assertion catches wrong Sionna modem conventions.
        tf.debugging.assert_equal(tf.shape(x_imp), tf.shape(x), message="RF OFDM roundtrip changed transmit-grid shape")

    meta.update({"backend": backend, "sample_rate_hz": float(sample_rate_hz)})
    return x_imp, meta


# -----------------------------------------------------------------------------
# Post-channel legacy stack kept for backward compatibility
# -----------------------------------------------------------------------------

def apply_realistic_rf_impairments_if_enabled(
    y: tf.Tensor,
    h: tf.Tensor | None,
    cfg: dict[str, Any],
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor | None]:
    """Backward-compatible post-channel hook.

    Realistic RF impairments are now applied to the transmit waveform before the
    existing channel call via ``apply_rf_impairments_to_transmit_grid_if_enabled``.
    This function intentionally does not modify y/h; it only preserves the old
    call site used by training/evaluation for the disabled legacy model.
    """
    del training
    return y, h


def apply_configured_impairments(
    y: tf.Tensor,
    h: tf.Tensor | None,
    cfg: dict[str, Any],
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor | None]:
    y, h = apply_realistic_rf_impairments_if_enabled(y, h, cfg, training=training)
    y, h = _apply_legacy_symbol_phase(y, h, cfg, training=training)
    return y, h


# Backward-compatible function name used by training/evaluation code.
def apply_symbol_phase_impairment(
    y: tf.Tensor,
    h: tf.Tensor | None,
    cfg: dict[str, Any],
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor | None]:
    return apply_configured_impairments(y, h, cfg, training=training)
