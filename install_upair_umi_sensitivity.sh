#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI}"
ROOT="$(cd "$ROOT" && pwd)"

if [[ "${UPAIR_ALLOW_ANY_ROOT:-0}" != "1" && "$ROOT" != *"_underUMI" ]]; then
  echo "[FATAL] Refusing to modify a directory that does not end in '_underUMI': $ROOT" >&2
  echo "        Set UPAIR_ALLOW_ANY_ROOT=1 only if this is intentional." >&2
  exit 2
fi

required=(
  "$ROOT/src/upair5g/builders.py"
  "$ROOT/src/upair5g/utils.py"
  "$ROOT/configs/twc_comprehensive_mu32_base.yaml"
  "$ROOT/scripts/run_isolated_eval_chunk.py"
  "$ROOT/scripts/isolated_eval_status.py"
  "$ROOT/scripts/merge_isolated_eval_chunks.py"
  "$ROOT/scripts/run_comprehensive_mu32_ablation.py"
  "$ROOT/upair_portable_env.sh"
)
for p in "${required[@]}"; do
  [[ -f "$p" ]] || { echo "[FATAL] Missing required file: $p" >&2; exit 2; }
done

cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

python - <<'PY_PREFLIGHT'
import inspect
from importlib.metadata import PackageNotFoundError, version

import tensorflow as tf
from sionna.phy.channel import OFDMChannel, gen_single_sector_topology
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, PanelArray, UMi

def package_version():
    for name in ("sionna-no-rt", "sionna"):
        try:
            return f"{name} {version(name)}"
        except PackageNotFoundError:
            pass
    return "unknown-distribution"

print("[PRECHECK] Python/TensorFlow =", tf.__version__)
print("[PRECHECK] Sionna =", package_version())
print("[PRECHECK] UMi =", inspect.signature(UMi))
print("[PRECHECK] OFDMChannel =", inspect.signature(OFDMChannel))
print("[PRECHECK] topology =", inspect.signature(gen_single_sector_topology))
assert issubclass(Antenna, PanelArray)
assert issubclass(AntennaArray, PanelArray)
PY_PREFLIGHT

mkdir -p "$ROOT/umi_sensitivity/backups" "$ROOT/logs/umi_sensitivity"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for p in src/upair5g/builders.py src/upair5g/utils.py; do
  backup="$ROOT/umi_sensitivity/backups/${p//\//__}.original"
  [[ -f "$backup" ]] || cp -p "$ROOT/$p" "$backup"
  cp -p "$ROOT/$p" "$ROOT/umi_sensitivity/backups/${p//\//__}.${stamp}"
done

cat > "$ROOT/src/upair5g/umi_channel.py" <<'PY_UMI'
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
PY_UMI

python - <<'PY_PATCH'
from pathlib import Path
import re

root = Path.cwd()

builders = root / "src/upair5g/builders.py"
text = builders.read_text(encoding="utf-8")
if "UPAIR_UMI_SENSITIVITY_DISPATCH_V1" not in text:
    pattern = re.compile(
        r'def build_channel\(cfg: dict\[str, Any\], tx: Any\) -> Any:\n'
        r'    num_tx = int\(first_present_attr\(tx, \["_upair_num_users", "num_tx", "_num_tx"\], 1\)\)\n'
        r'    if multiuser_enabled\(cfg\) and num_tx > 1:\n'
        r'        return _build_independent_multiuser_channel\(cfg, num_tx\)\n\n'
        r'    channel_model = _build_cdl_channel_model\(cfg\)\n'
        r'    return _build_ofdm_channel\(cfg, tx, channel_model, add_awgn=True\)\n'
    )
    replacement = """def build_channel(cfg: dict[str, Any], tx: Any) -> Any:
    # UPAIR_UMI_SENSITIVITY_DISPATCH_V1
    num_tx = int(first_present_attr(tx, ["_upair_num_users", "num_tx", "_num_tx"], 1))
    family = str(get_cfg(cfg, "channel.family", "cdl")).strip().lower().replace("-", "_")

    if family in {"umi", "urban_micro", "urban_microcell"}:
        from .umi_channel import build_dynamic_umi_ofdm_channel

        carrier_frequency = float(get_cfg(cfg, "pusch.carrier_frequency_hz"))
        channel_cfg = cfg["channel"]
        ut_array = _build_single_antenna(
            int(channel_cfg["num_tx_ant"]),
            carrier_frequency,
        )
        bs_array = _build_bs_array(
            int(channel_cfg["num_rx_ant"]),
            carrier_frequency,
        )
        return build_dynamic_umi_ofdm_channel(
            cfg,
            tx,
            num_users=num_tx,
            ut_array=ut_array,
            bs_array=bs_array,
        )

    if family not in {"cdl", "cdl_c", "cdlc"}:
        raise ValueError(
            f"Unsupported channel.family={family!r}. Supported values: cdl, umi."
        )

    if multiuser_enabled(cfg) and num_tx > 1:
        return _build_independent_multiuser_channel(cfg, num_tx)

    channel_model = _build_cdl_channel_model(cfg)
    return _build_ofdm_channel(cfg, tx, channel_model, add_awgn=True)
"""
    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(
            "[FATAL] Exact build_channel block was not found; no patch written."
        )
    builders.write_text(patched, encoding="utf-8")

utils = root / "src/upair5g/utils.py"
text = utils.read_text(encoding="utf-8")
if "UPAIR_SIONNA_GLOBAL_SEED_V1" not in text:
    old = """def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
"""
    new = """def set_global_seed(seed: int) -> None:
    # UPAIR_SIONNA_GLOBAL_SEED_V1
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        from sionna.phy import config as sionna_config
    except ImportError:
        from sionna import config as sionna_config
    sionna_config.seed = seed
"""
    if old not in text:
        raise SystemExit(
            "[FATAL] Exact set_global_seed block was not found; no patch written."
        )
    utils.write_text(text.replace(old, new, 1), encoding="utf-8")
PY_PATCH

python - <<'PY_CONFIG'
from copy import deepcopy
from pathlib import Path
import yaml

root = Path.cwd()
base = yaml.safe_load(
    (root / "configs/twc_comprehensive_mu32_base.yaml").read_text(
        encoding="utf-8"
    )
)
cfg = deepcopy(base)

cfg["experiment"]["name"] = "rx16_prb8_1dmrs_u3_umi_sensitivity"
cfg["system"]["ebno_db_eval"] = [-4, -3, -2, -1, 0, 1]
cfg["multiuser"]["eval_num_users"] = [3]
cfg["multiuser"]["fixed_num_users"] = 3

channel = cfg["channel"]
channel["family"] = "umi"
channel["model"] = "UMi"
channel["normalize_channel"] = True
channel["umi"] = {
    "profile_name": "umi_standard_topology_normalized",
    "scenario": "umi",
    "o2i_model": "low",
    "enable_pathloss": False,
    "enable_shadow_fading": False,
    "always_generate_lsp": False,
    "topology_resample": "per_channel_call",
    "min_bs_ut_dist_m": 10.0,
    "isd_m": 200.0,
    "bs_height_m": 10.0,
    "min_ut_height_m": 1.5,
    "max_ut_height_m": 1.5,
    "indoor_probability": 0.8,
    "min_speed_mps": float(channel["min_speed_mps"]),
    "max_speed_mps": float(channel["max_speed_mps"]),
    "force_los": None,
}

evaluation = cfg["evaluation"]
evaluation["receiver_microbatch_size"] = 8
evaluation["logical_batch_size"] = 64
evaluation["compiled_receiver_error_counts"] = False
evaluation["receiver_call_jit_compile"] = False
evaluation["stream_eval_microbatches"] = True
evaluation["min_num_batches_per_point"] = 20
evaluation["max_num_batches_per_point"] = 2000
evaluation["target_block_errors_per_receiver"] = 100
evaluation["reliable_min_block_errors"] = 100
evaluation["reliable_min_bit_errors"] = 1000
evaluation["nmse_receivers"] = []
evaluation["memory_cleanup_every_batches"] = 1
evaluation["memory_cleanup_every_microbatch"] = True

cfg["baselines"]["enabled_receivers"] = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]
cfg["baselines"]["covariance_estimation"]["reuse_cache"] = True
cfg["baselines"]["covariance_estimation"]["num_batches"] = 32
cfg["baselines"]["covariance_estimation"]["batch_size"] = 32
cfg["baselines"]["covariance_estimation"]["order"] = "f-t"
cfg["baselines"]["covariance_estimation"]["use_spatial_smoothing"] = False

path = root / "configs/twc_comprehensive_mu32_umi_sensitivity.yaml"
path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("[CONFIG] wrote", path)
PY_CONFIG

cat > "$ROOT/umi_sensitivity/driver.py" <<'PY_DRIVER'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CONFIG = ROOT / "configs" / "twc_comprehensive_mu32_umi_sensitivity.yaml"
PREFIX = "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]
BASELINES = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "perfect_csi_lmmse",
]
EBNOS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
TARGET = 100
MAX_BATCHES = 2000
MIN_BATCHES = 20
CHUNK_BATCHES = 20
MICRO = 8

CHECKPOINT_MANIFEST = (
    ROOT / "umi_sensitivity" / "checkpoint_manifest.sha256.json"
)
PROBE_MARKER = ROOT / "umi_sensitivity" / "PROBE_PASSED.json"
UPAIR_OUT = ROOT / "_umi_eval_chunks"
BASELINE_OUT = ROOT / "_umi_baseline_chunks"
COV_ROOT = ROOT / "_umi_shared_cov"
COV_NAME = "u3_prb8_umi_standard_topology_normalized"
COV_CACHE = COV_ROOT / COV_NAME / "artifacts" / "empirical_covariances.npz"


def sionna_version() -> str:
    for name in ("sionna-no-rt", "sionna"):
        try:
            return f"{name} {version(name)}"
        except PackageNotFoundError:
            continue
    return "unknown-distribution"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(variant: str) -> Path:
    return (
        ROOT
        / "TWC_plots_comprehensive"
        / "runs_rx16"
        / "seed7"
        / "1dmrs"
        / variant
        / "checkpoints"
        / "best.weights.h5"
    )


def train_state_path(variant: str) -> Path:
    return checkpoint_path(variant).parents[1] / "metrics" / "train_state.json"


def collect_checkpoint_manifest() -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        checkpoint = checkpoint_path(variant)
        state_path = train_state_path(variant)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing train state: {state_path}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        latest = int(state.get("latest_step", -1))
        total = int(state.get("total_steps", 40000))
        complete = bool(state.get("training_complete", False))
        if not complete or latest != 40000 or total != 40000:
            raise RuntimeError(
                f"{variant}: expected completed 40000-step training, got "
                f"complete={complete}, latest={latest}, total={total}."
            )

        variants[variant] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "latest_step": latest,
            "total_steps": total,
        }
    return {"variants": variants}


def guard_init() -> None:
    if CHECKPOINT_MANIFEST.exists():
        raise FileExistsError(
            f"{CHECKPOINT_MANIFEST} already exists; use guard-verify."
        )
    CHECKPOINT_MANIFEST.write_text(
        json.dumps(collect_checkpoint_manifest(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print("[GUARD] initialized", CHECKPOINT_MANIFEST)


def guard_verify() -> None:
    if not CHECKPOINT_MANIFEST.is_file():
        raise FileNotFoundError(CHECKPOINT_MANIFEST)
    expected = json.loads(
        CHECKPOINT_MANIFEST.read_text(encoding="utf-8")
    )
    current = collect_checkpoint_manifest()
    if current != expected:
        raise RuntimeError(
            "Checkpoint guard failed: one or more trained weights changed."
        )
    print("[GUARD] PASS: all 7 trained checkpoints are unchanged.")


def load_umi_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def static_probe() -> None:
    from sionna.phy.channel import OFDMChannel, gen_single_sector_topology
    from sionna.phy.channel.tr38901 import UMi
    from upair5g.builders import build_channel

    cfg = load_umi_config()
    channel = cfg["channel"]
    umi = channel["umi"]

    assert channel["family"] == "umi"
    assert channel["model"] == "UMi"
    assert channel["normalize_channel"] is True
    assert umi["profile_name"] == "umi_standard_topology_normalized"
    assert umi["scenario"] == "umi"
    assert umi["topology_resample"] == "per_channel_call"
    assert umi["enable_pathloss"] is False
    assert umi["enable_shadow_fading"] is False
    assert cfg["multiuser"]["eval_num_users"] == [3]
    assert cfg["system"]["ebno_db_eval"] == [-4, -3, -2, -1, 0, 1]
    assert "UPAIR_UMI_SENSITIVITY_DISPATCH_V1" in inspect.getsource(
        build_channel
    )

    print("[STATIC] Sionna =", sionna_version())
    print("[STATIC] UMi =", inspect.signature(UMi))
    print("[STATIC] OFDMChannel =", inspect.signature(OFDMChannel))
    print("[STATIC] topology =", inspect.signature(gen_single_sector_topology))
    print("[STATIC] profile =", umi["profile_name"])
    print("[STATIC] indoor_probability =", umi["indoor_probability"])
    print(
        "[STATIC] speed_mps =",
        umi["min_speed_mps"],
        umi["max_speed_mps"],
    )
    print("[STATIC] normalize_channel =", channel["normalize_channel"])
    print(
        "[STATIC] pathloss/shadow =",
        umi["enable_pathloss"],
        umi["enable_shadow_fading"],
    )
    guard_verify()
    print("[STATIC] PASS")


def point_rows(
    root: Path,
    *,
    variant: str,
    receiver: str,
    ebno_db: float,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not root.exists():
        return rows

    for path in root.rglob("chunk_result.csv"):
        try:
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            row = frame.iloc[0].to_dict()
            if str(row.get("variant", "")) != variant:
                continue
            if str(row.get("receiver", "")) != receiver:
                continue
            if int(row.get("num_users", -1)) != 3:
                continue
            if abs(float(row.get("ebno_db")) - float(ebno_db)) > 1e-9:
                continue
            index = int(row.get("chunk_idx", -1))
        except Exception:
            continue
        rows[index] = row
    return rows


def point_status(
    root: Path,
    *,
    variant: str,
    receiver: str,
    ebno_db: float,
) -> dict[str, Any]:
    rows = point_rows(
        root,
        variant=variant,
        receiver=receiver,
        ebno_db=ebno_db,
    )
    block_errors = sum(
        int(float(row.get("block_errors", 0) or 0))
        for row in rows.values()
    )
    num_blocks = sum(
        int(float(row.get("num_blocks", 0) or 0))
        for row in rows.values()
    )
    batches = sum(
        int(float(row.get("num_batches_run", 0) or 0))
        for row in rows.values()
    )
    done = (
        (batches >= MIN_BATCHES and block_errors >= TARGET)
        or batches >= MAX_BATCHES
    )
    next_chunk = -1
    if not done:
        for index in range(MAX_BATCHES // CHUNK_BATCHES):
            if index not in rows:
                next_chunk = index
                break
    return {
        "done": done,
        "block_errors": block_errors,
        "num_blocks": num_blocks,
        "batches": batches,
        "next_chunk": next_chunk,
        "num_chunks": len(rows),
        "bler": block_errors / num_blocks if num_blocks else float("nan"),
    }


def safe_tag(value: object) -> str:
    return (
        str(value)
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace(",", "_")
    )


def run_chunk(
    *,
    output_root: Path,
    variant: str,
    receiver: str,
    ebno_db: float,
    chunk_index: int,
    shared_covariance: Path | None = None,
) -> None:
    if shared_covariance is not None:
        tag = (
            f"{variant}_u3_{receiver}_"
            f"ebno{safe_tag(ebno_db)}_chunk{chunk_index:04d}_"
            f"m{MICRO}_b{CHUNK_BATCHES}"
        )
        cfg = load_umi_config()
        cache_name = str(
            cfg.get("baselines", {})
            .get("covariance_estimation", {})
            .get("cache_name", "empirical_covariances.npz")
        )
        target = output_root / tag / "artifacts" / cache_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared_covariance, target)
        print("[SHARED-COV] staged", shared_covariance, "->", target)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_isolated_eval_chunk.py"),
        "--config",
        str(CONFIG),
        "--variant",
        variant,
        "--dmrs-case",
        "1dmrs",
        "--seed",
        "7",
        "--num-users",
        "3",
        "--receiver",
        receiver,
        "--ebno-db",
        str(ebno_db),
        "--chunk-idx",
        str(chunk_index),
        "--chunk-batches",
        str(CHUNK_BATCHES),
        "--receiver-microbatch-size",
        str(MICRO),
        "--stageb-prefix",
        PREFIX,
        "--optuna-dir",
        str(ROOT / "optuna"),
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def merge_point(
    *,
    output_root: Path,
    variant: str,
    receiver: str,
    ebno_db: float,
) -> None:
    safe = str(ebno_db).replace("-", "m").replace(".", "p")
    output = output_root / f"merged_{variant}_u3_{receiver}_e{safe}.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "merge_isolated_eval_chunks.py"),
        "--input-root",
        str(output_root),
        "--output-csv",
        str(output),
        "--variant",
        variant,
        "--receiver",
        receiver,
        "--num-users",
        "3",
        "--ebno-db",
        str(ebno_db),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def require_probe() -> None:
    if not PROBE_MARKER.is_file():
        raise FileNotFoundError(
            "Missing umi_sensitivity/PROBE_PASSED.json. "
            "Run the mandatory GPU probe first."
        )


def eval_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    require_probe()
    guard_verify()
    UPAIR_OUT.mkdir(parents=True, exist_ok=True)

    for ebno_db in EBNOS:
        while True:
            status = point_status(
                UPAIR_OUT,
                variant=variant,
                receiver="upair5g_lmmse",
                ebno_db=ebno_db,
            )
            print(
                "[UMI-UPAIR]",
                variant,
                f"Eb/N0={ebno_db:g}",
                status,
                flush=True,
            )
            if status["done"]:
                break
            if status["next_chunk"] < 0:
                raise RuntimeError("No available next chunk index.")
            run_chunk(
                output_root=UPAIR_OUT,
                variant=variant,
                receiver="upair5g_lmmse",
                ebno_db=ebno_db,
                chunk_index=int(status["next_chunk"]),
            )
        merge_point(
            output_root=UPAIR_OUT,
            variant=variant,
            receiver="upair5g_lmmse",
            ebno_db=ebno_db,
        )

    guard_verify()
    print("[UMI-UPAIR] COMPLETE", variant)


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_shared_covariance() -> None:
    from scripts.run_comprehensive_mu32_ablation import (
        _apply_optuna_best_1dmrs,
        _eval_cfg,
        _variant_cfg,
    )
    from upair5g.baselines import estimate_empirical_covariances
    from upair5g.builders import build_channel, build_pusch_transmitter
    from upair5g.config import ensure_output_tree, load_config, set_cfg
    from upair5g.utils import set_global_seed

    base = load_config(CONFIG)
    train_cfg = _variant_cfg(base, "main_d256_b4_r2", "1dmrs", 7)
    _apply_optuna_best_1dmrs(
        train_cfg,
        "main_d256_b4_r2",
        "1dmrs",
        storage_dir=ROOT / "optuna",
        study_prefix=PREFIX,
        require_external=True,
    )
    cfg = _eval_cfg(train_cfg, "main_d256_b4_r2", "1dmrs", 3)
    set_cfg(cfg, "system.seed", 7007)
    set_cfg(cfg, "system.evaluation_seed", 7007)
    set_cfg(cfg, "multiuser.fixed_num_users", 3)
    set_cfg(cfg, "experiment.output_root", str(COV_ROOT))
    set_cfg(cfg, "experiment.name", COV_NAME)
    set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", True)
    set_cfg(
        cfg,
        "baselines.covariance_estimation.cache_name",
        "empirical_covariances.npz",
    )

    payload = {
        "channel": cfg["channel"],
        "pusch": cfg["pusch"],
        "multiuser_dmrs": cfg["multiuser"]["dmrs"],
        "num_users": 3,
        "covariance": cfg["baselines"]["covariance_estimation"],
        "seed": 7007,
        "sionna_version": sionna_version(),
    }
    expected_fingerprint = fingerprint(payload)
    manifest_path = COV_CACHE.parent / "manifest.json"

    if COV_CACHE.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == expected_fingerprint:
            print("[UMI-COV] reuse", COV_CACHE)
            return

    profile_root = COV_ROOT / COV_NAME
    if profile_root.exists():
        shutil.rmtree(profile_root)

    set_global_seed(7007)
    paths = ensure_output_tree(cfg)
    tx, _ = build_pusch_transmitter(cfg, num_users=3)
    channel = build_channel(cfg, tx)
    result = estimate_empirical_covariances(
        tx=tx,
        channel=channel,
        cfg=cfg,
        paths=paths,
    )
    cache_value = result["cache_path"]
    if hasattr(cache_value, "numpy"):
        raw = cache_value.numpy()
        actual_cache = Path(
            raw.decode() if isinstance(raw, bytes) else str(raw)
        )
    else:
        actual_cache = Path(str(cache_value))

    if actual_cache.resolve() != COV_CACHE.resolve():
        raise RuntimeError(
            f"Unexpected covariance path: {actual_cache} != {COV_CACHE}"
        )

    manifest = {
        "cache": str(COV_CACHE),
        "fingerprint": expected_fingerprint,
        "fingerprint_payload": payload,
        "sionna_version": sionna_version(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[UMI-COV] wrote", COV_CACHE)


def eval_baselines() -> None:
    require_probe()
    guard_verify()
    build_shared_covariance()
    BASELINE_OUT.mkdir(parents=True, exist_ok=True)

    for receiver in BASELINES:
        for ebno_db in EBNOS:
            while True:
                status = point_status(
                    BASELINE_OUT,
                    variant="main_d256_b4_r2",
                    receiver=receiver,
                    ebno_db=ebno_db,
                )
                print(
                    "[UMI-BASELINE]",
                    receiver,
                    f"Eb/N0={ebno_db:g}",
                    status,
                    flush=True,
                )
                if status["done"]:
                    break
                if status["next_chunk"] < 0:
                    raise RuntimeError("No available next chunk index.")
                run_chunk(
                    output_root=BASELINE_OUT,
                    variant="main_d256_b4_r2",
                    receiver=receiver,
                    ebno_db=ebno_db,
                    chunk_index=int(status["next_chunk"]),
                    shared_covariance=(
                        COV_CACHE
                        if receiver == "baseline_ls_2dlmmse_lmmse"
                        else None
                    ),
                )
            merge_point(
                output_root=BASELINE_OUT,
                variant="main_d256_b4_r2",
                receiver=receiver,
                ebno_db=ebno_db,
            )

    guard_verify()
    print("[UMI-BASELINE] COMPLETE")


def status_report() -> bool:
    all_done = True
    print("UMI UPAIR+LMMSE, U=3")
    for variant in VARIANTS:
        cells = []
        for ebno_db in EBNOS:
            status = point_status(
                UPAIR_OUT,
                variant=variant,
                receiver="upair5g_lmmse",
                ebno_db=ebno_db,
            )
            mark = "D" if status["done"] else "M"
            all_done = all_done and bool(status["done"])
            cells.append(
                f"{ebno_db:+g}:{mark}"
                f"({status['block_errors']}/{status['batches']})"
            )
        print(f"{variant:28s} " + "  ".join(cells))

    print("\nUMI BASELINES, U=3")
    for receiver in BASELINES:
        cells = []
        for ebno_db in EBNOS:
            status = point_status(
                BASELINE_OUT,
                variant="main_d256_b4_r2",
                receiver=receiver,
                ebno_db=ebno_db,
            )
            mark = "D" if status["done"] else "M"
            all_done = all_done and bool(status["done"])
            cells.append(
                f"{ebno_db:+g}:{mark}"
                f"({status['block_errors']}/{status['batches']})"
            )
        print(f"{receiver:36s} " + "  ".join(cells))

    print(
        "\nLegend: D=100 block errors reached or 2000-batch cap; "
        "M=incomplete/missing."
    )
    print(f"OVERALL_COMPLETE={int(all_done)}")
    return all_done


def audit_outputs() -> None:
    guard_verify()
    count = 0
    for output_root in (UPAIR_OUT, BASELINE_OUT):
        if not output_root.exists():
            continue
        for result_path in output_root.rglob("chunk_result.csv"):
            frame = pd.read_csv(result_path)
            if frame.empty:
                raise RuntimeError(f"Empty result: {result_path}")
            row = frame.iloc[0]
            if int(row["num_users"]) != 3:
                raise RuntimeError(f"Wrong user count: {result_path}")
            if not any(
                abs(float(row["ebno_db"]) - value) < 1e-9
                for value in EBNOS
            ):
                raise RuntimeError(f"Wrong Eb/N0: {result_path}")

            resolved = (
                result_path.parent / "artifacts" / "resolved_config.yaml"
            )
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            cfg = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            assert cfg["channel"]["family"] == "umi"
            assert cfg["channel"]["model"] == "UMi"
            assert cfg["channel"]["normalize_channel"] is True
            assert (
                cfg["channel"]["umi"]["profile_name"]
                == "umi_standard_topology_normalized"
            )
            count += 1
    print(
        f"[AUDIT] PASS: {count} chunks have the intended UMi resolved config."
    )


def _runtime_sample(seed: int, batch_size: int) -> dict[str, Any]:
    import tensorflow as tf
    from upair5g.builders import (
        build_channel,
        build_pusch_transmitter,
        get_resource_grid,
    )
    from upair5g.config import set_cfg
    from upair5g.utils import (
        call_channel,
        call_transmitter,
        ebno_db_to_no,
        set_global_seed,
    )

    cfg = copy.deepcopy(load_umi_config())
    set_cfg(cfg, "system.seed", seed)
    set_cfg(cfg, "system.evaluation_seed", seed)
    set_cfg(cfg, "multiuser.fixed_num_users", 3)
    set_global_seed(seed)

    tx, _ = build_pusch_transmitter(cfg, num_users=3)
    channel = build_channel(cfg, tx)
    x, bits = call_transmitter(tx, batch_size)
    no = ebno_db_to_no(
        tf.constant(-2.0, tf.float32),
        tx=tx,
        resource_grid=get_resource_grid(tx),
    )
    y, h = call_channel(channel, x, no)

    if getattr(channel, "_upair_channel_family", None) != "umi":
        raise RuntimeError("Channel dispatch did not create the UMi wrapper.")
    if tuple(x.shape[:3]) != (batch_size, 3, 1):
        raise RuntimeError(f"Unexpected x shape: {x.shape}")
    if tuple(y.shape[:3]) != (batch_size, 1, 16):
        raise RuntimeError(f"Unexpected y shape: {y.shape}")
    if tuple(h.shape[:5]) != (batch_size, 1, 16, 3, 1):
        raise RuntimeError(f"Unexpected h shape: {h.shape}")
    if tuple(h.shape[-2:]) != tuple(x.shape[-2:]):
        raise RuntimeError(f"Resource-grid mismatch: x={x.shape}, h={h.shape}")

    y_np = np.asarray(y)
    h_np = np.asarray(h)
    if not np.isfinite(y_np).all() or not np.isfinite(h_np).all():
        raise RuntimeError("NaN/Inf found in UMi tensors.")

    link_power = tf.reduce_mean(
        tf.square(tf.abs(h)),
        axis=(2, 4, 5, 6),
    ).numpy()
    power_error = float(np.max(np.abs(link_power - 1.0)))
    if power_error >= 5e-4:
        raise RuntimeError(
            f"Per-link normalization failed; max error={power_error}."
        )

    summary = channel.last_topology_summary
    if summary["min_speed_mps"] < 8.33 - 1e-4:
        raise RuntimeError(summary)
    if summary["max_speed_mps"] > 16.67 + 1e-4:
        raise RuntimeError(summary)

    return {
        "x": np.asarray(x),
        "bits": None if bits is None else np.asarray(bits),
        "y": y_np,
        "h": h_np,
        "summary": summary,
        "power_error": power_error,
    }


def runtime_probe() -> None:
    import tensorflow as tf
    from scripts.run_comprehensive_mu32_ablation import (
        _apply_optuna_best_1dmrs,
        _eval_cfg,
        _variant_cfg,
    )
    from upair5g.builders import (
        build_channel,
        build_ls_estimator,
        build_pusch_transmitter,
        extract_true_dmrs_mask_per_stream,
        get_resource_grid,
    )
    from upair5g.config import get_cfg, load_config, set_cfg
    from upair5g.estimator import UPAIRChannelEstimator
    from upair5g.evaluation import _make_eval_batch, evaluate_model
    from upair5g.utils import set_global_seed

    guard_verify()

    for batch_size in (1, 4, 8, 32):
        result = _runtime_sample(12000 + batch_size, batch_size)
        print(
            f"[RUNTIME] B={batch_size} "
            f"x={result['x'].shape} y={result['y'].shape} "
            f"h={result['h'].shape} "
            f"power_error={result['power_error']:.3e} "
            f"topology={result['summary']}"
        )
        del result
        tf.keras.backend.clear_session()
        gc.collect()

    first = _runtime_sample(777, 4)
    tf.keras.backend.clear_session()
    gc.collect()
    second = _runtime_sample(777, 4)
    tf.keras.backend.clear_session()
    gc.collect()
    third = _runtime_sample(778, 4)

    if not np.array_equal(first["bits"], second["bits"]):
        raise RuntimeError("Same-seed transmitted bits are not repeatable.")
    if not np.allclose(first["h"], second["h"], rtol=0.0, atol=1e-6):
        raise RuntimeError("Same-seed UMi channel is not repeatable.")
    if not np.allclose(first["y"], second["y"], rtol=0.0, atol=1e-6):
        raise RuntimeError("Same-seed received signal is not repeatable.")
    if np.allclose(first["h"], third["h"], rtol=0.0, atol=1e-6):
        raise RuntimeError("Different seeds produced the same UMi channel.")
    print("[RUNTIME] reproducibility PASS")

    smoke_root = ROOT / "_umi_smoke_end_to_end"
    shutil.rmtree(smoke_root, ignore_errors=True)
    base = load_config(CONFIG)
    train_cfg = _variant_cfg(base, "main_d256_b4_r2", "1dmrs", 7)
    _apply_optuna_best_1dmrs(
        train_cfg,
        "main_d256_b4_r2",
        "1dmrs",
        storage_dir=ROOT / "optuna",
        study_prefix=PREFIX,
        require_external=True,
    )
    cfg = _eval_cfg(train_cfg, "main_d256_b4_r2", "1dmrs", 3)
    set_cfg(cfg, "experiment.output_root", str(smoke_root))
    set_cfg(cfg, "experiment.name", "all_receivers")
    set_cfg(cfg, "system.ebno_db_eval", [-2.0])
    set_cfg(cfg, "system.batch_size_eval", 4)
    set_cfg(cfg, "evaluation.logical_batch_size", 4)
    set_cfg(cfg, "evaluation.receiver_microbatch_size", 2)
    set_cfg(cfg, "evaluation.min_num_batches_per_point", 1)
    set_cfg(cfg, "evaluation.max_num_batches_per_point", 1)
    set_cfg(cfg, "evaluation.target_block_errors_per_receiver", 0)
    set_cfg(cfg, "evaluation.per_receiver_stopping", False)
    set_cfg(cfg, "evaluation.force", True)
    set_cfg(cfg, "evaluation.save_example_batch", False)
    set_cfg(cfg, "evaluation.compiled_receiver_error_counts", False)
    set_cfg(cfg, "evaluation.receiver_call_jit_compile", False)
    set_cfg(
        cfg,
        "baselines.enabled_receivers",
        [
            "baseline_ls_lmmse",
            "baseline_ls_2dlmmse_lmmse",
            "upair5g_lmmse",
            "perfect_csi_lmmse",
        ],
    )
    set_cfg(
        cfg,
        "evaluation.nmse_receivers",
        [
            "baseline_ls_lmmse",
            "baseline_ls_2dlmmse_lmmse",
            "upair5g_lmmse",
            "perfect_csi_lmmse",
        ],
    )
    set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", False)
    set_cfg(cfg, "baselines.covariance_estimation.num_batches", 2)
    set_cfg(cfg, "baselines.covariance_estimation.batch_size", 4)

    checkpoint = checkpoint_path("main_d256_b4_r2")
    result = evaluate_model(
        cfg,
        checkpoint_path=str(checkpoint),
        num_users=3,
    )
    curves = pd.read_csv(result["curves_path"])
    expected_receivers = {
        "baseline_ls_lmmse",
        "baseline_ls_2dlmmse_lmmse",
        "upair5g_lmmse",
        "perfect_csi_lmmse",
    }
    if set(curves["receiver"]) != expected_receivers or len(curves) != 4:
        raise RuntimeError(curves)
    if not np.isfinite(curves["bler"].to_numpy(float)).all():
        raise RuntimeError("Non-finite smoke-test BLER.")
    if not np.isfinite(curves["nmse"].to_numpy(float)).all():
        raise RuntimeError("Non-finite smoke-test NMSE.")
    print(
        curves[
            [
                "receiver",
                "bler",
                "nmse",
                "receiver_ms_per_batch",
            ]
        ].to_string(index=False)
    )
    print("[RUNTIME] end-to-end all-receiver smoke PASS")

    for index, variant in enumerate(VARIANTS):
        train_cfg = _variant_cfg(base, variant, "1dmrs", 7)
        _apply_optuna_best_1dmrs(
            train_cfg,
            variant,
            "1dmrs",
            storage_dir=ROOT / "optuna",
            study_prefix=PREFIX,
            require_external=True,
        )
        cfg = _eval_cfg(train_cfg, variant, "1dmrs", 3)
        seed = 20000 + index
        set_cfg(cfg, "system.seed", seed)
        set_cfg(cfg, "system.evaluation_seed", seed)
        set_global_seed(seed)

        tx, _ = build_pusch_transmitter(cfg, num_users=3)
        channel = build_channel(cfg, tx)
        resource_grid = get_resource_grid(tx)
        pilot_mask = extract_true_dmrs_mask_per_stream(tx, resource_grid)
        ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
        estimator = UPAIRChannelEstimator(
            ls_estimator=ls_estimator,
            resource_grid=resource_grid,
            cfg=cfg,
            pilot_mask=pilot_mask,
        )
        batch = _make_eval_batch(
            tx,
            channel,
            cfg,
            batch_size=1,
            ebno_db=-2.0,
        )
        estimator.estimate_with_ls(
            batch["y"],
            batch["no"],
            training=False,
            ls_estimator=ls_estimator,
            pilot_mask=pilot_mask,
        )
        estimator.load_weights(str(checkpoint_path(variant)))
        h_hat, err_hat, _, _ = estimator.estimate_with_ls(
            batch["y"],
            batch["no"],
            training=False,
            ls_estimator=ls_estimator,
            pilot_mask=pilot_mask,
        )
        if tuple(h_hat.shape) != tuple(batch["h"].shape):
            raise RuntimeError(
                f"{variant}: output shape mismatch "
                f"{h_hat.shape} != {batch['h'].shape}"
            )
        if not np.isfinite(np.asarray(h_hat)).all():
            raise RuntimeError(f"{variant}: non-finite h_hat.")
        if not np.isfinite(np.asarray(err_hat)).all():
            raise RuntimeError(f"{variant}: non-finite err_hat.")
        print("[RUNTIME] checkpoint forward PASS", variant)

        del estimator, batch, channel, tx, ls_estimator, h_hat, err_hat
        tf.keras.backend.clear_session()
        gc.collect()

    guard_verify()
    PROBE_MARKER.write_text(
        json.dumps(
            {
                "passed": True,
                "utc": datetime.now(timezone.utc).isoformat(),
                "config": str(CONFIG.relative_to(ROOT)),
                "profile": "umi_standard_topology_normalized",
                "sionna": sionna_version(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[RUNTIME] ALL PROBES PASS")
    print("[RUNTIME] wrote", PROBE_MARKER)


def signature(kind: str, variant: str | None) -> None:
    root = UPAIR_OUT if kind == "upair" else BASELINE_OUT
    count = 0
    for path in root.rglob("chunk_result.csv") if root.exists() else []:
        try:
            frame = pd.read_csv(path, nrows=1)
            if frame.empty:
                continue
            row = frame.iloc[0]
            if int(row.get("num_users", -1)) != 3:
                continue
            if kind == "upair":
                if str(row.get("variant", "")) != variant:
                    continue
                if str(row.get("receiver", "")) != "upair5g_lmmse":
                    continue
            else:
                if str(row.get("receiver", "")) not in BASELINES:
                    continue
            count += 1
        except Exception:
            pass
    print(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("guard-init")
    sub.add_parser("guard-verify")
    sub.add_parser("static")
    sub.add_parser("probe-runtime")

    variant_parser = sub.add_parser("eval-variant")
    variant_parser.add_argument("variant")

    sub.add_parser("eval-baselines")
    sub.add_parser("status")
    sub.add_parser("audit")

    signature_parser = sub.add_parser("signature")
    signature_parser.add_argument("--kind", choices=["upair", "baseline"], required=True)
    signature_parser.add_argument("--variant", default=None)

    args = parser.parse_args()

    if args.command == "guard-init":
        guard_init()
    elif args.command == "guard-verify":
        guard_verify()
    elif args.command == "static":
        static_probe()
    elif args.command == "probe-runtime":
        runtime_probe()
    elif args.command == "eval-variant":
        eval_variant(args.variant)
    elif args.command == "eval-baselines":
        eval_baselines()
    elif args.command == "status":
        guard_verify()
        status_report()
    elif args.command == "audit":
        audit_outputs()
    elif args.command == "signature":
        signature(args.kind, args.variant)


if __name__ == "__main__":
    main()
PY_DRIVER

cat > "$ROOT/umi_sensitivity/probe_2h.sbatch" <<'SH_PROBE'
#!/usr/bin/env bash
#SBATCH --job-name=umi-probe
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/umi_sensitivity/%x_%j.out

set -euo pipefail
ROOT="${UPAIR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" && -f "$ROOT/upair_portable_env.sh" ]] || {
  echo "[FATAL] Invalid repository root: $ROOT" >&2
  exit 2
}
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

rm -rf "$ROOT/_umi_smoke_end_to_end"
python "$ROOT/umi_sensitivity/driver.py" static
python "$ROOT/umi_sensitivity/driver.py" probe-runtime
SH_PROBE

cat > "$ROOT/umi_sensitivity/variant_24h.sbatch" <<'SH_VARIANT'
#!/usr/bin/env bash
#SBATCH --job-name=umi-upair
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/umi_sensitivity/%x_%j.out

set -euo pipefail
VARIANT="${1:?Usage: sbatch umi_sensitivity/variant_24h.sbatch <variant>}"
ROOT="${UPAIR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" && -f "$ROOT/upair_portable_env.sh" ]] || {
  echo "[FATAL] Invalid repository root: $ROOT" >&2
  exit 2
}
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

last="-1"
stagnant=0
while true; do
  set +e
  python "$ROOT/umi_sensitivity/driver.py" eval-variant "$VARIANT"
  rc=$?
  set -e
  [[ $rc -eq 0 ]] && exit 0

  now="$(python "$ROOT/umi_sensitivity/driver.py" signature --kind upair --variant "$VARIANT")"
  echo "[24H] variant=$VARIANT rc=$rc chunks=$now"
  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last="$now"
  if (( stagnant >= 2 )); then
    echo "[FATAL] No UMi evaluation progress across three attempts." >&2
    exit "$rc"
  fi
  sleep 15
done
SH_VARIANT

cat > "$ROOT/umi_sensitivity/baselines_24h.sbatch" <<'SH_BASELINES'
#!/usr/bin/env bash
#SBATCH --job-name=umi-baselines
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/umi_sensitivity/%x_%j.out

set -euo pipefail
ROOT="${UPAIR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" && -f "$ROOT/upair_portable_env.sh" ]] || {
  echo "[FATAL] Invalid repository root: $ROOT" >&2
  exit 2
}
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_VMODULE="${TF_CPP_VMODULE:-bfc_allocator=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

last="-1"
stagnant=0
while true; do
  set +e
  python "$ROOT/umi_sensitivity/driver.py" eval-baselines
  rc=$?
  set -e
  [[ $rc -eq 0 ]] && exit 0

  now="$(python "$ROOT/umi_sensitivity/driver.py" signature --kind baseline)"
  echo "[24H] baselines rc=$rc chunks=$now"
  if [[ "$now" == "$last" ]]; then
    stagnant=$((stagnant + 1))
  else
    stagnant=0
  fi
  last="$now"
  if (( stagnant >= 2 )); then
    echo "[FATAL] No UMi baseline progress across three attempts." >&2
    exit "$rc"
  fi
  sleep 15
done
SH_BASELINES

cat > "$ROOT/umi_sensitivity/submit_probe.sh" <<'SH_SUBMIT_PROBE'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate
python "$ROOT/umi_sensitivity/driver.py" static
mkdir -p "$ROOT/logs/umi_sensitivity"
sbatch --export=ALL,UPAIR_REPO_ROOT="$ROOT" "$ROOT/umi_sensitivity/probe_2h.sbatch"
SH_SUBMIT_PROBE

cat > "$ROOT/umi_sensitivity/submit_full.sh" <<'SH_SUBMIT_FULL'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/upair_portable_env.sh"
upair_activate

[[ -f "$ROOT/umi_sensitivity/PROBE_PASSED.json" ]] || {
  echo "[FATAL] Missing umi_sensitivity/PROBE_PASSED.json." >&2
  echo "        Run and inspect the mandatory GPU probe first." >&2
  exit 2
}
python "$ROOT/umi_sensitivity/driver.py" static
mkdir -p "$ROOT/logs/umi_sensitivity"

variants=(
  main_d256_b4_r2
  shallow_d256_b2_r2
  deep_d256_b6_r2
  narrow_d192_b4_r2
  wide_d320_b4_r2
  wide_deep_d320_b6_r2
  mlpwide_d256_b4_r4
)

for variant in "${variants[@]}"; do
  sbatch \
    --export=ALL,UPAIR_REPO_ROOT="$ROOT" \
    --job-name="umi-${variant:0:18}" \
    "$ROOT/umi_sensitivity/variant_24h.sbatch" "$variant"
done

sbatch \
  --export=ALL,UPAIR_REPO_ROOT="$ROOT" \
  "$ROOT/umi_sensitivity/baselines_24h.sbatch"
SH_SUBMIT_FULL

cat > "$ROOT/umi_sensitivity/README.txt" <<'TXT_README'
UMi sensitivity profile
=======================
Evaluation only. No training function is called.

Channel:
- Sionna UMi, uplink, one BS, three UTs.
- Fresh single-sector topology for every channel call.
- Standard UMi geometry: 10 m minimum BS-UT distance, 200 m ISD,
  10 m BS height, 1.5 m UT height, 0.8 indoor probability.
- Mobility 8.33--16.67 m/s, matching CDL-C training.
- Same one-antenna UT and 1x16 omni BS array as training.
- LoS/NLoS sampled according to UMi; not forced.
- Pathloss and shadow fading disabled.
- Per-link resource-grid normalization enabled.

Interpretation:
This is a normalized small-scale channel-mismatch test. It is not a
link-budget, power-control, near-far, pathloss, or shadow-fading test.

Evaluation:
- U=3
- Eb/N0=-4,-3,-2,-1,0,+1 dB
- 100 block errors or 2000 logical batches
- 20 batches/chunk
- receiver microbatch 8
- seven UPAIR jobs plus one benchmark job

Separate outputs:
- _umi_eval_chunks
- _umi_baseline_chunks
- _umi_shared_cov
- logs/umi_sensitivity

The seven trained checkpoints are protected by a SHA-256 manifest.
TXT_README

chmod +x \
  "$ROOT/umi_sensitivity/driver.py" \
  "$ROOT/umi_sensitivity/probe_2h.sbatch" \
  "$ROOT/umi_sensitivity/variant_24h.sbatch" \
  "$ROOT/umi_sensitivity/baselines_24h.sbatch" \
  "$ROOT/umi_sensitivity/submit_probe.sh" \
  "$ROOT/umi_sensitivity/submit_full.sh"

python -m py_compile \
  "$ROOT/src/upair5g/umi_channel.py" \
  "$ROOT/src/upair5g/builders.py" \
  "$ROOT/src/upair5g/utils.py" \
  "$ROOT/umi_sensitivity/driver.py"

bash -n \
  "$ROOT/umi_sensitivity/probe_2h.sbatch" \
  "$ROOT/umi_sensitivity/variant_24h.sbatch" \
  "$ROOT/umi_sensitivity/baselines_24h.sbatch" \
  "$ROOT/umi_sensitivity/submit_probe.sh" \
  "$ROOT/umi_sensitivity/submit_full.sh"

if [[ -f "$ROOT/umi_sensitivity/checkpoint_manifest.sha256.json" ]]; then
  python "$ROOT/umi_sensitivity/driver.py" guard-verify
else
  python "$ROOT/umi_sensitivity/driver.py" guard-init
fi

python "$ROOT/umi_sensitivity/driver.py" static

echo
echo "[OK] Installed UMi sensitivity support in:"
echo "     $ROOT/umi_sensitivity"
echo
echo "[NEXT] Submit only the mandatory probe:"
echo "       bash umi_sensitivity/submit_probe.sh"
echo
echo "Do not submit the full evaluation until the probe log ends with:"
echo "       [RUNTIME] ALL PROBES PASS"

