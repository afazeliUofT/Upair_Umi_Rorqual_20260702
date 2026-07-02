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
