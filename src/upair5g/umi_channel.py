from __future__ import annotations

from typing import Any

import tensorflow as tf

from .compat import first_present_attr, instantiate_filtered, resolve_attr, safe_call_variants
from .config import get_cfg


def _resource_grid(tx: Any) -> Any:
    resource_grid = first_present_attr(tx, ["resource_grid", "_resource_grid"], None)
    if resource_grid is None:
        raise AttributeError("Could not locate resource_grid in PUSCH transmitter.")
    return resource_grid


def _cfg_float(
    cfg: dict[str, Any],
    path: str,
    fallback: float | None = None,
) -> float | None:
    value = get_cfg(cfg, path, fallback)
    return None if value is None else float(value)


def _scalar_float(value: Any) -> float:
    return float(tf.convert_to_tensor(value).numpy())


class DynamicUMiOFDMChannel:
    """UMi OFDM channel with a fresh topology matching every input batch."""

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        tx: Any,
        num_users: int,
        ut_array: Any,
        bs_array: Any,
    ) -> None:
        self.cfg = cfg
        self.num_users = int(num_users)
        self.return_channel = True
        self._upair_channel_family = "umi"
        self._upair_topology_resample_per_call = True
        self._upair_call_count = 0
        self._upair_last_topology_summary: dict[str, Any] = {}

        UMi = resolve_attr(
            ["sionna.phy.channel.tr38901", "sionna.channel.tr38901"],
            "UMi",
        )
        OFDMChannel = resolve_attr(
            ["sionna.phy.channel", "sionna.channel"],
            "OFDMChannel",
        )
        self._topology_generator = resolve_attr(
            ["sionna.phy.channel", "sionna.channel"],
            "gen_single_sector_topology",
        )

        carrier_frequency = float(get_cfg(cfg, "pusch.carrier_frequency_hz"))
        precision = str(get_cfg(cfg, "system.precision", "single"))

        self.channel_model = instantiate_filtered(
            UMi,
            carrier_frequency=carrier_frequency,
            o2i_model=str(get_cfg(cfg, "channel.umi.o2i_model", "low")),
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            enable_pathloss=bool(
                get_cfg(cfg, "channel.umi.enable_pathloss", False)
            ),
            enable_shadow_fading=bool(
                get_cfg(cfg, "channel.umi.enable_shadow_fading", False)
            ),
            always_generate_lsp=bool(
                get_cfg(cfg, "channel.umi.always_generate_lsp", False)
            ),
            precision=precision,
        )

        self.ofdm_channel = instantiate_filtered(
            OFDMChannel,
            channel_model=self.channel_model,
            resource_grid=_resource_grid(tx),
            normalize_channel=bool(
                get_cfg(cfg, "channel.normalize_channel", True)
            ),
            return_channel=True,
            precision=precision,
        )

        force_los = get_cfg(cfg, "channel.umi.force_los", None)
        self._force_los = None if force_los is None else bool(force_los)
        self._topology_kwargs = {
            "num_ut": self.num_users,
            "scenario": str(get_cfg(cfg, "channel.umi.scenario", "umi")),
            "min_bs_ut_dist": _cfg_float(
                cfg, "channel.umi.min_bs_ut_dist_m", 10.0
            ),
            "isd": _cfg_float(cfg, "channel.umi.isd_m", 200.0),
            "bs_height": _cfg_float(cfg, "channel.umi.bs_height_m", 10.0),
            "min_ut_height": _cfg_float(
                cfg, "channel.umi.min_ut_height_m", 1.5
            ),
            "max_ut_height": _cfg_float(
                cfg, "channel.umi.max_ut_height_m", 1.5
            ),
            "indoor_probability": _cfg_float(
                cfg, "channel.umi.indoor_probability", 0.8
            ),
            "min_ut_velocity": _cfg_float(
                cfg,
                "channel.umi.min_speed_mps",
                float(get_cfg(cfg, "channel.min_speed_mps", 0.0)),
            ),
            "max_ut_velocity": _cfg_float(
                cfg,
                "channel.umi.max_speed_mps",
                float(get_cfg(cfg, "channel.max_speed_mps", 0.0)),
            ),
            "precision": precision,
        }

    @property
    def last_topology_summary(self) -> dict[str, Any]:
        return dict(self._upair_last_topology_summary)

    def _set_fresh_topology(self, batch_size: int) -> None:
        topology = self._topology_generator(
            batch_size=int(batch_size),
            **{
                key: value
                for key, value in self._topology_kwargs.items()
                if value is not None
            },
        )
        self.channel_model.set_topology(*topology, los=self._force_los)

        ut_loc, bs_loc, _, _, velocities, in_state = topology
        delta = ut_loc - bs_loc[:, :1, :]
        distance_2d = tf.sqrt(
            tf.reduce_sum(tf.square(delta[..., :2]), axis=-1)
        )
        speed = tf.sqrt(
            tf.reduce_sum(tf.square(velocities), axis=-1)
        )

        scenario = first_present_attr(self.channel_model, ["_scenario"], None)
        los_fraction = float("nan")
        if scenario is not None:
            los = first_present_attr(scenario, ["los"], None)
            if los is not None:
                los_fraction = _scalar_float(
                    tf.reduce_mean(tf.cast(los, tf.float32))
                )

        self._upair_call_count += 1
        self._upair_last_topology_summary = {
            "call_index": self._upair_call_count,
            "batch_size": int(batch_size),
            "num_users": self.num_users,
            "indoor_fraction": _scalar_float(
                tf.reduce_mean(tf.cast(in_state, tf.float32))
            ),
            "los_fraction": los_fraction,
            "min_distance_2d_m": _scalar_float(tf.reduce_min(distance_2d)),
            "max_distance_2d_m": _scalar_float(tf.reduce_max(distance_2d)),
            "min_speed_mps": _scalar_float(tf.reduce_min(speed)),
            "max_speed_mps": _scalar_float(tf.reduce_max(speed)),
        }

    def __call__(
        self,
        x: tf.Tensor,
        no: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if not tf.executing_eagerly():
            raise RuntimeError(
                "Dynamic UMi topology must be generated in eager mode."
            )

        x = tf.convert_to_tensor(x)
        if x.shape.rank != 5:
            raise ValueError(
                f"Expected x=[B,U,S,T,F], got shape={x.shape}."
            )

        static_users = x.shape[1]
        if static_users is not None and int(static_users) != self.num_users:
            raise ValueError(
                f"UMi model expects {self.num_users} users, got {static_users}."
            )

        batch_size = (
            int(x.shape[0])
            if x.shape[0] is not None
            else int(tf.shape(x)[0].numpy())
        )
        self._set_fresh_topology(batch_size)

        try:
            output = self.ofdm_channel(x, no)
        except (tf.errors.ResourceExhaustedError, MemoryError):
            raise
        except Exception:
            output = safe_call_variants(self.ofdm_channel, x, no)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("UMi OFDM channel did not return (y, h).")
        return tf.convert_to_tensor(output[0]), tf.convert_to_tensor(output[1])


def build_dynamic_umi_ofdm_channel(
    cfg: dict[str, Any],
    tx: Any,
    *,
    num_users: int,
    ut_array: Any,
    bs_array: Any,
) -> DynamicUMiOFDMChannel:
    return DynamicUMiOFDMChannel(
        cfg=cfg,
        tx=tx,
        num_users=num_users,
        ut_array=ut_array,
        bs_array=bs_array,
    )
