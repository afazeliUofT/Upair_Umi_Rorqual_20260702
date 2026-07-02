from __future__ import annotations

from typing import Any

import tensorflow as tf

from .builders import extract_pilot_mask_per_stream
from .compat import safe_call_variants
from .utils import (
    btfnu_to_tensor7,
    broadcast_like_err,
    broadcast_no_feature,
    btfnc_to_tensor7,
    complex_to_ri_channels,
    pad_user_dim,
    tensor7_to_btfnu,
    tensor7_to_btfnc,
    y_to_btfnc,
)


class FiLMAxialBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_rate = float(dropout)

        self.norm0 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.freq_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.d_model // self.num_heads),
            dropout=self.dropout_rate,
        )
        self.time_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.d_model // self.num_heads),
            dropout=self.dropout_rate,
        )

        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same")
        self.pwconv = tf.keras.layers.Conv2D(self.d_model, kernel_size=1, padding="same")
        self.fc1 = tf.keras.layers.Dense(int(self.d_model * self.mlp_ratio))
        self.fc2 = tf.keras.layers.Dense(self.d_model)
        self.prompt_proj = tf.keras.layers.Dense(2 * self.d_model)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.activation = tf.keras.layers.Activation("gelu")

    def _film(self, x: tf.Tensor, prompt: tf.Tensor) -> tf.Tensor:
        gamma, beta = tf.split(self.prompt_proj(prompt), 2, axis=-1)
        gamma = gamma[:, tf.newaxis, tf.newaxis, :]
        beta = beta[:, tf.newaxis, tf.newaxis, :]
        return x * (1.0 + gamma) + beta

    def call(self, x: tf.Tensor, prompt: tf.Tensor, training: bool = False) -> tf.Tensor:
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        f = tf.shape(x)[2]
        d = tf.shape(x)[3]

        z = self._film(self.norm0(x), prompt)
        zf = tf.reshape(z, [b * t, f, d])
        af = self.freq_attn(zf, zf, training=training)
        af = self.dropout(af, training=training)
        af = tf.reshape(af, [b, t, f, d])
        x = x + af

        z = self._film(self.norm1(x), prompt)
        zt = tf.transpose(z, [0, 2, 1, 3])
        zt = tf.reshape(zt, [b * f, t, d])
        at = self.time_attn(zt, zt, training=training)
        at = self.dropout(at, training=training)
        at = tf.reshape(at, [b, f, t, d])
        at = tf.transpose(at, [0, 2, 1, 3])
        x = x + at

        z = self._film(self.norm2(x), prompt)
        lc = self.dwconv(z)
        lc = self.pwconv(lc)
        lc = self.activation(lc)
        lc = self.dropout(lc, training=training)
        x = x + lc

        z = self._film(self.norm3(x), prompt)
        mlp = self.fc1(z)
        mlp = self.activation(mlp)
        mlp = self.dropout(mlp, training=training)
        mlp = self.fc2(mlp)
        mlp = self.dropout(mlp, training=training)
        x = x + mlp

        return x


class UPAIRChannelEstimator(tf.keras.Model):
    def __init__(
        self,
        ls_estimator: Any,
        resource_grid: Any,
        cfg: dict[str, Any],
        pilot_mask: tf.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="upair_channel_estimator", **kwargs)
        self.cfg = cfg
        self.ls_estimator = ls_estimator
        self.num_rx_ant = int(cfg["channel"]["num_rx_ant"])
        self.max_num_users = int(cfg.get("multiuser", {}).get("max_num_users", 1))
        self.d_model = int(cfg["model"]["d_model"])
        model_cfg = cfg.get("model", {})
        self.use_noise_feature = bool(model_cfg.get("use_noise_feature", True))
        self.use_pilot_mask_feature = bool(model_cfg.get("use_pilot_mask_feature", True))
        self.pilot_mask_mode = str(model_cfg.get("pilot_mask_mode", "per_stream")).lower()
        self.error_feature_mode = str(model_cfg.get("error_feature_mode", "per_user")).lower()
        self.residual_scale = float(model_cfg.get("residual_scale", 0.35))
        self.eps = 1e-6

        per_stream_mask = self.pilot_mask_mode in {"per_stream", "per_user", "stream", "user"}
        per_user_error = self.error_feature_mode in {"per_user", "per_stream", "user", "stream"}
        self.pilot_mask_channels = self.max_num_users if (self.use_pilot_mask_feature and per_stream_mask) else int(self.use_pilot_mask_feature)
        self.error_feature_channels = self.max_num_users if per_user_error else 1

        extra_channels = self.error_feature_channels + int(self.use_noise_feature) + self.pilot_mask_channels
        if self.max_num_users > 1:
            input_channels = 2 * self.num_rx_ant * self.max_num_users + 2 * self.num_rx_ant + extra_channels
        else:
            input_channels = 4 * self.num_rx_ant + extra_channels

        self.stem = tf.keras.layers.Conv2D(self.d_model, kernel_size=1, padding="same")
        self.prompt_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.d_model, activation="gelu"),
                tf.keras.layers.Dense(self.d_model),
            ]
        )
        self.blocks = [
            FiLMAxialBlock(
                d_model=self.d_model,
                num_heads=int(cfg["model"]["num_heads"]),
                mlp_ratio=float(cfg["model"]["mlp_ratio"]),
                dropout=float(cfg["model"]["dropout"]),
                name=f"axial_block_{i}",
            )
            for i in range(int(cfg["model"]["num_blocks"]))
        ]

        # Start exactly from LS and learn only residual corrections.
        self.delta_head = tf.keras.layers.Conv2D(
            2 * self.num_rx_ant * self.max_num_users,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="delta_head",
        )
        self.err_head = tf.keras.layers.Conv2D(
            self.num_rx_ant * self.max_num_users,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="err_head",
        )

        self.pilot_mask = tf.cast(
            tf.convert_to_tensor(pilot_mask) if pilot_mask is not None else extract_pilot_mask_per_stream(resource_grid),
            tf.float32,
        )
        self.input_channels = input_channels

    def _finalize_build_after_direct_forward(self) -> None:
        """
        Keras flips ``model.built`` when ``Model.__call__`` is used.
        Our training/evaluation paths directly invoke ``estimate_with_ls()``,
        which creates the variables but bypasses that flag update.
        Mark the model built once the first direct forward has succeeded so
        checkpoint save/load works with Keras 3.
        """
        if not self.built and self.weights:
            self.built = True

    def _parse_inputs(self, inputs: Any, *args: Any) -> tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, (tuple, list)):
            if len(inputs) < 2:
                raise ValueError("Expected at least y and no as inputs.")
            y, no = inputs[0], inputs[1]
        elif len(args) >= 1:
            y, no = inputs, args[0]
        else:
            raise ValueError("Could not parse estimator inputs.")
        return tf.convert_to_tensor(y), tf.convert_to_tensor(no)

    def _call_ls(self, y: tf.Tensor, no: tf.Tensor, ls_estimator: Any | None = None) -> tuple[tf.Tensor, tf.Tensor]:
        estimator = ls_estimator or self.ls_estimator
        try:
            out = estimator(y, no)
        except (tf.errors.ResourceExhaustedError, MemoryError):
            raise
        except Exception:
            out = safe_call_variants(estimator, y, no)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError("LS estimator must return (h_hat, err_var).")
        return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])

    def _pad_feature_dim(self, x: tf.Tensor, target_channels: int) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        if x.shape.rank != 4:
            raise ValueError(f"Expected rank-4 feature map [B,T,F,C], got rank {x.shape.rank}.")
        target_channels = int(target_channels)
        pad_channels = tf.maximum(target_channels - tf.shape(x)[-1], 0)
        paddings = tf.stack(
            [
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([tf.constant(0, dtype=tf.int32), tf.cast(pad_channels, tf.int32)]),
            ]
        )
        return tf.pad(x, paddings)[..., :target_channels]

    def _pad_mask_streams(self, mask: tf.Tensor, target_streams: int) -> tf.Tensor:
        mask = tf.convert_to_tensor(mask)
        if mask.shape.rank != 3:
            raise ValueError(f"Expected rank-3 pilot mask [T,F,S], got rank {mask.shape.rank}.")
        target_streams = int(target_streams)
        pad_streams = tf.maximum(target_streams - tf.shape(mask)[-1], 0)
        paddings = tf.stack(
            [
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([tf.constant(0, dtype=tf.int32), tf.cast(pad_streams, tf.int32)]),
            ]
        )
        return tf.pad(mask, paddings)[..., :target_streams]

    def _pilot_mask_for_batch(
        self,
        pilot_mask: tf.Tensor | None,
        batch: tf.Tensor,
        time: tf.Tensor,
        freq: tf.Tensor,
        *,
        collapse: bool = False,
    ) -> tf.Tensor:
        mask = tf.cast(tf.convert_to_tensor(pilot_mask if pilot_mask is not None else self.pilot_mask), tf.float32)
        if mask.shape.rank == 2:
            mask = mask[..., tf.newaxis]
        if mask.shape.rank != 3:
            raise ValueError(f"Expected pilot mask rank 2 or 3, got {mask.shape.rank}.")

        per_stream = self.pilot_mask_mode in {"per_stream", "per_user", "stream", "user"}
        if collapse or not per_stream:
            mask = tf.reduce_max(mask, axis=-1, keepdims=True)
            channels = 1
        else:
            mask = self._pad_mask_streams(mask, self.max_num_users)
            channels = self.max_num_users
        return tf.broadcast_to(mask[tf.newaxis, ...], [batch, time, freq, channels])

    def _build_features(
        self,
        y: tf.Tensor,
        h_ls: tf.Tensor,
        err_ls: tf.Tensor,
        no: tf.Tensor,
        pilot_mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        y_btfnc = y_to_btfnc(y)

        err_bc = broadcast_like_err(err_ls, h_ls)
        per_user_error = self.error_feature_mode in {"per_user", "per_stream", "user", "stream"}
        if self.max_num_users > 1:
            h_ls_btfnu = tensor7_to_btfnu(h_ls)
            err_btfnu = tensor7_to_btfnu(err_bc)
            h_feat = pad_user_dim(h_ls_btfnu, self.max_num_users)
            b = tf.shape(y_btfnc)[0]
            t = tf.shape(y_btfnc)[1]
            f = tf.shape(y_btfnc)[2]
            h_ri = complex_to_ri_channels(tf.reshape(h_feat, [b, t, f, self.num_rx_ant * self.max_num_users]))
            if per_user_error:
                # Sionna's err_var can have batch dimension 1 even when y/h have
                # batch dimension B>1. Use the broadcasted error tensor err_btfnu,
                # not raw err_ls, so every feature has the same [B,T,F,*] axes.
                err_user_map = tf.reduce_mean(err_btfnu, axis=-2)
                err_map = self._pad_feature_dim(err_user_map, self.max_num_users)
            else:
                err_feat = pad_user_dim(err_btfnu, self.max_num_users)
                err_map = tf.reduce_mean(err_feat, axis=[-2, -1], keepdims=False)[..., tf.newaxis]
        else:
            h_ls_btfnu = tensor7_to_btfnu(h_ls)
            err_btfnu = tensor7_to_btfnu(err_bc)
            h_ls_btfnc = tf.squeeze(h_ls_btfnu, axis=-1)
            h_ri = complex_to_ri_channels(h_ls_btfnc)
            if per_user_error:
                # Use broadcasted err_btfnu for the same reason as in the
                # multi-user branch: raw err_ls may carry batch dimension 1.
                err_map = tf.reduce_mean(err_btfnu, axis=-2)
            else:
                err_btfnc = tf.squeeze(err_btfnu, axis=-1)
                err_map = tf.reduce_mean(err_btfnc, axis=-1, keepdims=True)

        features = [
            h_ri,
            complex_to_ri_channels(y_btfnc),
            err_map,
        ]

        b = tf.shape(y_btfnc)[0]
        t = tf.shape(y_btfnc)[1]
        f = tf.shape(y_btfnc)[2]

        if self.use_noise_feature:
            features.append(broadcast_no_feature(no, b, t, f))
        if self.use_pilot_mask_feature:
            features.append(self._pilot_mask_for_batch(pilot_mask, b, t, f))

        feat = tf.concat(features, axis=-1)
        return feat, h_ls_btfnu, err_btfnu, y_btfnc

    def _compute_prompt(self, z: tf.Tensor, pilot_mask: tf.Tensor | None = None) -> tf.Tensor:
        b = tf.shape(z)[0]
        t = tf.shape(z)[1]
        f = tf.shape(z)[2]
        mask = self._pilot_mask_for_batch(pilot_mask, b, t, f, collapse=True)
        denom = tf.reduce_sum(mask, axis=[1, 2], keepdims=False) + 1e-6
        pooled = tf.reduce_sum(z * mask, axis=[1, 2], keepdims=False) / denom
        return self.prompt_mlp(pooled)

    def estimate_with_ls(
        self,
        y: tf.Tensor,
        no: tf.Tensor,
        training: bool = False,
        ls_estimator: Any | None = None,
        pilot_mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        h_ls, err_ls = self._call_ls(y, no, ls_estimator=ls_estimator)
        feat, h_ls_btfnu, err_btfnu, _ = self._build_features(y, h_ls, err_ls, no, pilot_mask=pilot_mask)

        z = self.stem(feat)
        prompt = self._compute_prompt(z, pilot_mask=pilot_mask)

        for block in self.blocks:
            z = block(z, prompt, training=training)

        delta = self.delta_head(z)
        err_delta = self.err_head(z)
        b = tf.shape(z)[0]
        t = tf.shape(z)[1]
        f = tf.shape(z)[2]
        delta = tf.reshape(delta, [b, t, f, self.num_rx_ant, self.max_num_users, 2])
        real_delta, imag_delta = tf.unstack(delta, axis=-1)

        residual = tf.complex(real_delta, imag_delta)
        h_anchor = pad_user_dim(h_ls_btfnu, self.max_num_users)
        err_anchor = pad_user_dim(err_btfnu, self.max_num_users)
        h_hat_btfnu = h_anchor + tf.cast(self.residual_scale, residual.dtype) * residual
        err_delta = tf.reshape(err_delta, [b, t, f, self.num_rx_ant, self.max_num_users])
        # Multiplicative positive correction.  With the zero-initialized err_head,
        # this starts exactly from the LS error variance instead of adding
        # softplus(-4) to it.
        err_scale = tf.exp(tf.clip_by_value(err_delta, -6.0, 6.0))
        err_hat_btfnu = tf.maximum(err_anchor * err_scale, tf.cast(self.eps, err_anchor.dtype))

        actual_users = tf.shape(h_ls_btfnu)[-1]
        h_hat_btfnu = h_hat_btfnu[..., :actual_users]
        err_hat_btfnu = err_hat_btfnu[..., :actual_users]

        h_hat = btfnu_to_tensor7(h_hat_btfnu)
        err_hat = tf.cast(btfnu_to_tensor7(err_hat_btfnu), tf.float32)

        self._finalize_build_after_direct_forward()

        return h_hat, err_hat, h_ls, err_ls

    def call(self, inputs: Any, *args: Any, training: bool = False, **kwargs: Any) -> tuple[tf.Tensor, tf.Tensor]:
        y, no = self._parse_inputs(inputs, *args)
        h_hat, err_hat, _, _ = self.estimate_with_ls(y, no, training=training)
        return h_hat, err_hat


class UPAIRChannelEstimatorView(tf.keras.layers.Layer):
    """K-specific view over a shared UPAIR model.

    This lets comprehensive evaluation build separate Sionna receivers for
    1, 2, 3, and 4 scheduled users while loading one shared set of UPAIR
    weights trained with random user-count sampling.
    """

    def __init__(
        self,
        estimator: UPAIRChannelEstimator,
        ls_estimator: Any,
        pilot_mask: tf.Tensor,
        name: str = "upair_channel_estimator_view",
        **kwargs: Any,
    ) -> None:
        super().__init__(trainable=False, name=name, **kwargs)
        self.estimator = estimator
        self.ls_estimator = ls_estimator
        self.pilot_mask = tf.cast(pilot_mask, tf.float32)

    def call(self, inputs: Any, *args: Any, training: bool = False, **kwargs: Any) -> tuple[tf.Tensor, tf.Tensor]:
        del kwargs
        y, no = self.estimator._parse_inputs(inputs, *args)
        h_hat, err_hat, _, _ = self.estimator.estimate_with_ls(
            y,
            no,
            training=training,
            ls_estimator=self.ls_estimator,
            pilot_mask=self.pilot_mask,
        )
        return h_hat, err_hat
