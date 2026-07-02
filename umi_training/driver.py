#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.optuna_1dmrs_common import SUGGESTED_PARAM_NAMES  # noqa: E402
from scripts.run_comprehensive_mu32_ablation import (  # noqa: E402
    _apply_optuna_best_1dmrs,
    _eval_cfg,
    _variant_cfg,
)
from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.training import train_model  # noqa: E402

UMI_CONFIG = ROOT / "configs" / "twc_comprehensive_mu32_umi_training.yaml"
CDLC_CONFIG = ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"

VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]

CDLC_STAGEB_PREFIX = (
    "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
)
UMI_STAGEA_PREFIX = (
    "umiNorm_v1_b32_prb8_u34610_1dmrs_stageA"
)
UMI_STAGEB_PREFIX = (
    "umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"
)

STAGE_TOTAL = {"A": 20, "B": 6}
STAGE_STEPS = {"A": 4000, "B": 12000}
STAGE_SOURCE_TOP_K = {"A": 0, "B": 6}

DB_ROOT = ROOT / "UMI_training" / "optuna_db"
RUN_ROOT = ROOT / "UMI_training" / "runs_rx16" / "seed7" / "1dmrs"
SMOKE_ROOT = ROOT / "UMI_training" / "smoke" / "seed7" / "1dmrs"
FINAL_MANIFEST = ROOT / "UMI_training" / "trained_checkpoint_manifest.json"

UMI_EVAL_ROOT = ROOT / "_umi_trained_umi_eval_chunks"
CDLC_EVAL_ROOT = ROOT / "_umi_trained_cdlc_eval_chunks"

TARGET_ERRORS = 100
MIN_BATCHES = 20
MAX_BATCHES = 2000
CHUNK_BATCHES = 20
EBNOS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha() -> str:
    return sha256(UMI_CONFIG)


def stage_prefix(stage: str) -> str:
    stage = stage.upper()
    if stage == "A":
        return UMI_STAGEA_PREFIX
    if stage == "B":
        return UMI_STAGEB_PREFIX
    raise ValueError(stage)


def study_name(stage: str, variant: str) -> str:
    return f"{stage_prefix(stage)}_{variant}"


def db_path(stage: str, variant: str) -> Path:
    return DB_ROOT / f"{study_name(stage, variant)}.db"


def storage_url(stage: str, variant: str) -> str:
    return f"sqlite:///{db_path(stage, variant).resolve()}"


def study_manifest_path(stage: str, variant: str) -> Path:
    return DB_ROOT / f"{study_name(stage, variant)}.manifest.json"


def best_json_path(stage: str, variant: str) -> Path:
    return ROOT / "optuna" / f"{study_name(stage, variant)}_best_params.json"


def old_best_json_path(variant: str) -> Path:
    return ROOT / "optuna" / f"{CDLC_STAGEB_PREFIX}_{variant}_best_params.json"


def final_output_dir(variant: str) -> Path:
    return RUN_ROOT / variant


def final_checkpoint(variant: str) -> Path:
    return final_output_dir(variant) / "checkpoints" / "best.weights.h5"


def final_state_path(variant: str) -> Path:
    return final_output_dir(variant) / "metrics" / "train_state.json"


def validate_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def original_guard() -> None:
    probe = ROOT / "umi_sensitivity" / "PROBE_PASSED.json"
    if not probe.is_file():
        raise FileNotFoundError(
            "Missing umi_sensitivity/PROBE_PASSED.json. "
            "The UMi runtime probe must pass before UMi training."
        )
    marker = load_json(probe)
    if not bool(marker.get("passed", False)):
        raise RuntimeError(f"Invalid UMi probe marker: {probe}")

    command = [
        sys.executable,
        str(ROOT / "umi_sensitivity" / "driver.py"),
        "guard-verify",
    ]
    import subprocess
    subprocess.run(command, cwd=ROOT, check=True)


def static_check() -> None:
    if not UMI_CONFIG.is_file():
        raise FileNotFoundError(UMI_CONFIG)
    cfg = load_config(UMI_CONFIG)
    assert get_cfg(cfg, "channel.family") == "umi"
    assert get_cfg(cfg, "channel.model") == "UMi"
    assert bool(get_cfg(cfg, "channel.normalize_channel")) is True
    assert (
        get_cfg(cfg, "channel.umi.profile_name")
        == "umi_standard_topology_normalized"
    )
    assert get_cfg(cfg, "system.ebno_db_eval") == [-4, -3, -2, -1, 0, 1]
    assert int(get_cfg(cfg, "training.steps")) == 40000
    assert RUN_ROOT.resolve() != (
        ROOT
        / "TWC_plots_comprehensive"
        / "runs_rx16"
        / "seed7"
        / "1dmrs"
    ).resolve()

    for variant in VARIANTS:
        path = old_best_json_path(variant)
        if not path.is_file():
            raise FileNotFoundError(path)

    original_guard()
    print("[STATIC] UMi training config:", UMI_CONFIG)
    print("[STATIC] UMi final output root:", RUN_ROOT)
    print("[STATIC] Stage A:", STAGE_TOTAL["A"], "trials x", STAGE_STEPS["A"], "steps")
    print("[STATIC] Stage B:", STAGE_TOTAL["B"], "trials x", STAGE_STEPS["B"], "steps")
    print("[STATIC] final training: 40000 steps from scratch")
    print("[STATIC] PASS")


def _write_or_verify_study_manifest(stage: str, variant: str) -> None:
    stage = stage.upper()
    payload = {
        "stage": stage,
        "variant": variant,
        "study_name": study_name(stage, variant),
        "storage": storage_url(stage, variant),
        "config": str(UMI_CONFIG.relative_to(ROOT)),
        "config_sha256": config_sha(),
        "channel_family": "umi",
        "profile": "umi_standard_topology_normalized",
        "target_total_trials": STAGE_TOTAL[stage],
        "steps": STAGE_STEPS[stage],
        "source_top_k": STAGE_SOURCE_TOP_K[stage],
    }
    if stage == "A":
        payload["warm_start_json"] = str(old_best_json_path(variant).relative_to(ROOT))
    else:
        payload["source_study"] = study_name("A", variant)
        payload["source_storage"] = storage_url("A", variant)

    path = study_manifest_path(stage, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = load_json(path)
        if current != payload:
            raise RuntimeError(
                f"Study manifest mismatch at {path}. "
                "Refusing to mix different tuning configurations."
            )
    else:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def prepare_study(stage: str, variant: str) -> None:
    stage = stage.upper()
    validate_variant(variant)
    original_guard()
    _write_or_verify_study_manifest(stage, variant)

    if stage == "B":
        a = get_study_status("A", variant)
        if not a["ready"]:
            raise RuntimeError(
                f"Stage A is not ready for {variant}: {a}"
            )

    DB_ROOT.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name(stage, variant),
        storage=storage_url(stage, variant),
        direction="minimize",
        load_if_exists=True,
    )

    if stage == "A":
        payload = load_json(old_best_json_path(variant))
        source = dict(payload.get("best_params", {}))
        required_names = {
            "learning_rate_schedule",
            "learning_rate",
            "weight_decay",
            "nmse_loss_weight",
            "grad_clip_norm",
            "dropout",
            "residual_scale",
        }
        schedule = str(source.get("learning_rate_schedule", "cosine_decay"))
        candidate_names = set(required_names)
        if schedule in {"cosine_decay", "polynomial_decay"}:
            candidate_names.update(
                {
                    "learning_rate_decay_fraction",
                    "learning_rate_final_fraction",
                }
            )
        if schedule == "polynomial_decay":
            candidate_names.add("learning_rate_polynomial_power")
        candidate = {
            name: source[name]
            for name in candidate_names
            if name in source
        }
        missing = required_names - set(candidate)
        if missing:
            raise RuntimeError(
                f"Old best JSON for {variant} lacks required fields: {sorted(missing)}"
            )
        try:
            study.enqueue_trial(
                candidate,
                user_attrs={
                    "source_domain": "cdl_c",
                    "purpose": "UMi Stage-A warm start",
                },
                skip_if_exists=True,
            )
        except TypeError:
            existing = [
                dict(t.params) for t in study.get_trials(deepcopy=False)
                if t.params
            ]
            if candidate not in existing:
                study.enqueue_trial(candidate)

    print(
        f"[PREPARE] stage={stage} variant={variant} "
        f"study={study_name(stage, variant)} storage={storage_url(stage, variant)}"
    )


def get_study_status(stage: str, variant: str) -> dict[str, Any]:
    stage = stage.upper()
    validate_variant(variant)
    path = db_path(stage, variant)
    if not path.is_file():
        return {
            "exists": False,
            "finished": 0,
            "completed": 0,
            "pruned": 0,
            "failed": 0,
            "running": 0,
            "target": STAGE_TOTAL[stage],
            "ready": False,
            "best_value": math.nan,
        }

    study = optuna.load_study(
        study_name=study_name(stage, variant),
        storage=storage_url(stage, variant),
    )
    counts = {state.name: 0 for state in optuna.trial.TrialState}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1

    finished = counts.get("COMPLETE", 0) + counts.get("PRUNED", 0)
    completed = counts.get("COMPLETE", 0)
    ready = (
        finished >= STAGE_TOTAL[stage]
        and completed >= (6 if stage == "A" else 1)
        and best_json_path(stage, variant).is_file()
    )
    best_value = float(study.best_value) if completed else math.nan
    return {
        "exists": True,
        "finished": finished,
        "completed": completed,
        "pruned": counts.get("PRUNED", 0),
        "failed": counts.get("FAIL", 0),
        "running": counts.get("RUNNING", 0),
        "waiting": counts.get("WAITING", 0),
        "target": STAGE_TOTAL[stage],
        "ready": ready,
        "best_value": best_value,
    }


def print_study_status(stage: str) -> int:
    stage = stage.upper()
    all_ready = True
    for index, variant in enumerate(VARIANTS):
        status = get_study_status(stage, variant)
        all_ready &= bool(status["ready"])
        print(
            f"{index} {variant:28s} "
            f"finished={status['finished']:2d}/{status['target']:2d} "
            f"complete={status['completed']:2d} "
            f"pruned={status['pruned']:2d} "
            f"failed={status['failed']:2d} "
            f"running={status['running']:2d} "
            f"best={status['best_value']:.6g} "
            f"ready={int(status['ready'])}"
        )
    print(f"STAGE_{stage}_COMPLETE={int(all_ready)}")
    return 0 if all_ready else 1


def pending_indices_for_stage(stage: str) -> str:
    pending = [
        str(index)
        for index, variant in enumerate(VARIANTS)
        if not get_study_status(stage, variant)["ready"]
    ]
    return ",".join(pending)


def _apply_final_common(cfg: dict[str, Any], *, steps: int) -> None:
    set_cfg(cfg, "training.steps", int(steps))
    set_cfg(cfg, "training.resume", True)
    set_cfg(cfg, "training.checkpoint_every", min(1000, int(steps)))
    set_cfg(cfg, "training.log_every", min(100, int(steps)))
    set_cfg(cfg, "training.eval_every", min(2000, int(steps)))
    set_cfg(cfg, "training.val_steps", 96)
    set_cfg(cfg, "training.val_ebno_db", [-4.0, -2.0, 0.0, 2.0, 4.0])
    set_cfg(cfg, "training.val_user_counts", [1, 2, 3, 4])
    set_cfg(cfg, "training.val_user_count_weights", [1.0, 3.0, 6.0, 10.0])
    set_cfg(cfg, "training.val_microbatch_size", 16)
    set_cfg(cfg, "system.batch_size_train", 32)
    set_cfg(cfg, "system.batch_size_eval", 32)
    set_cfg(cfg, "system.seed", 7)
    set_cfg(cfg, "system.training_seed", 7)
    set_cfg(cfg, "system.evaluation_seed", 1007)
    set_cfg(cfg, "multiuser.fixed_num_users", 3)
    set_cfg(cfg, "experiment.training_domain", "umi")
    set_cfg(cfg, "experiment.channel_profile", "umi_standard_topology_normalized")


def make_training_cfg(
    variant: str,
    *,
    steps: int,
    output_root: Path,
    use_umi_tuned_params: bool,
    resume: bool,
) -> dict[str, Any]:
    validate_variant(variant)
    base = load_config(UMI_CONFIG)
    cfg = _variant_cfg(base, variant, "1dmrs", 7)
    _apply_final_common(cfg, steps=steps)

    prefix = UMI_STAGEB_PREFIX if use_umi_tuned_params else CDLC_STAGEB_PREFIX
    _apply_optuna_best_1dmrs(
        cfg,
        variant,
        "1dmrs",
        storage_dir=ROOT / "optuna",
        study_prefix=prefix,
        require_external=True,
    )

    # Reassert output/training controls after the compatibility loader.
    _apply_final_common(cfg, steps=steps)
    set_cfg(cfg, "training.resume", bool(resume))
    set_cfg(cfg, "experiment.output_root", str(output_root))
    set_cfg(cfg, "experiment.name", variant)
    return cfg


def smoke_train(variant: str) -> None:
    validate_variant(variant)
    original_guard()
    output = SMOKE_ROOT / variant
    if output.exists():
        import shutil
        shutil.rmtree(output)

    cfg = make_training_cfg(
        variant,
        steps=200,
        output_root=SMOKE_ROOT,
        use_umi_tuned_params=False,
        resume=False,
    )
    set_cfg(cfg, "training.eval_every", 100)
    set_cfg(cfg, "training.checkpoint_every", 100)
    set_cfg(cfg, "training.log_every", 20)
    set_cfg(cfg, "training.val_steps", 12)
    set_cfg(cfg, "training.val_microbatch_size", 8)
    set_cfg(cfg, "system.batch_size_eval", 16)

    result = train_model(cfg)
    if not bool(result.get("training_complete", False)):
        raise RuntimeError(f"Smoke training incomplete for {variant}: {result}")

    history = load_json(Path(result["history_path"])).get("history", [])
    validation = [
        row for row in history
        if isinstance(row, dict) and "val_nmse_prop" in row
    ]
    if not validation:
        raise RuntimeError(f"No validation rows for smoke run {variant}")
    for row in validation:
        for key in ("val_nmse_prop", "val_nmse_ls", "loss"):
            if not np.isfinite(float(row[key])):
                raise RuntimeError(
                    f"Non-finite {key} in smoke run {variant}: {row[key]}"
                )
    print(
        f"[SMOKE] PASS {variant} "
        f"last_val_nmse={float(validation[-1]['val_nmse_prop']):.6g}"
    )
    original_guard()


def smoke_status() -> int:
    all_complete = True
    for index, variant in enumerate(VARIANTS):
        state = SMOKE_ROOT / variant / "metrics" / "train_state.json"
        complete = False
        step = -1
        finite = False
        if state.is_file():
            payload = load_json(state)
            complete = bool(payload.get("training_complete", False))
            step = int(payload.get("latest_step", -1))
            history_path = SMOKE_ROOT / variant / "metrics" / "history.json"
            if history_path.is_file():
                rows = load_json(history_path).get("history", [])
                vals = [
                    float(row["val_nmse_prop"])
                    for row in rows
                    if isinstance(row, dict) and "val_nmse_prop" in row
                ]
                finite = bool(vals) and all(np.isfinite(vals))
        passed = complete and step == 200 and finite
        all_complete &= passed
        print(
            f"{index} {variant:28s} step={step:4d}/200 "
            f"complete={int(complete)} finite_validation={int(finite)} "
            f"pass={int(passed)}"
        )
    print(f"UMI_TRAINING_SMOKE_COMPLETE={int(all_complete)}")
    return 0 if all_complete else 1


def _verify_stage_b_ready(variant: str) -> None:
    status = get_study_status("B", variant)
    if not status["ready"]:
        raise RuntimeError(
            f"UMi Stage B is not complete for {variant}: {status}"
        )
    manifest = study_manifest_path("B", variant)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    payload = load_json(manifest)
    if payload.get("config_sha256") != config_sha():
        raise RuntimeError(f"UMi Stage-B config fingerprint mismatch for {variant}")


def train_final(variant: str) -> None:
    validate_variant(variant)
    original_guard()
    _verify_stage_b_ready(variant)

    state_path = final_state_path(variant)
    if state_path.is_file():
        state = load_json(state_path)
        if (
            bool(state.get("training_complete", False))
            and int(state.get("latest_step", -1)) == 40000
            and final_checkpoint(variant).is_file()
        ):
            print(f"[FINAL-TRAIN] already complete: {variant}")
            original_guard()
            return

    cfg = make_training_cfg(
        variant,
        steps=40000,
        output_root=RUN_ROOT,
        use_umi_tuned_params=True,
        resume=True,
    )
    result = train_model(cfg)
    if not bool(result.get("training_complete", False)):
        print(
            f"[FINAL-TRAIN] incomplete {variant}: "
            f"{result.get('latest_step')}/{result.get('total_steps')}"
        )
        raise SystemExit(75)

    checkpoint = Path(result["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(
        f"[FINAL-TRAIN] COMPLETE {variant} "
        f"checkpoint={checkpoint} sha256={sha256(checkpoint)}"
    )
    original_guard()


def final_training_status() -> int:
    all_complete = True
    for index, variant in enumerate(VARIANTS):
        state_path = final_state_path(variant)
        complete = False
        step = -1
        reason = "missing"
        best_val = math.nan
        checkpoint = final_checkpoint(variant)
        if state_path.is_file():
            payload = load_json(state_path)
            complete = bool(payload.get("training_complete", False))
            step = int(payload.get("latest_step", -1))
            reason = str(payload.get("save_reason", ""))
            best_val = float(payload.get("best_val", math.nan))
        passed = complete and step == 40000 and checkpoint.is_file()
        all_complete &= passed
        print(
            f"{index} {variant:28s} step={step:5d}/40000 "
            f"complete={int(complete)} checkpoint={int(checkpoint.is_file())} "
            f"best_val={best_val:.6g} reason={reason}"
        )
    print(f"UMI_FINAL_TRAINING_COMPLETE={int(all_complete)}")
    return 0 if all_complete else 1


def _current_final_manifest_payload() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for variant in VARIANTS:
        checkpoint = final_checkpoint(variant)
        state_path = final_state_path(variant)
        if not checkpoint.is_file() or not state_path.is_file():
            continue
        state = load_json(state_path)
        if not (
            bool(state.get("training_complete", False))
            and int(state.get("latest_step", -1)) == 40000
        ):
            continue
        rows[variant] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "latest_step": int(state.get("latest_step", -1)),
            "best_val": float(state.get("best_val", math.nan)),
            "training_domain": "umi",
            "channel_profile": "umi_standard_topology_normalized",
            "stage_b_best_json": str(
                best_json_path("B", variant).relative_to(ROOT)
            ),
        }
    return {
        "config": str(UMI_CONFIG.relative_to(ROOT)),
        "config_sha256": config_sha(),
        "variants": rows,
    }


def _write_final_manifest(payload: dict[str, Any]) -> None:
    FINAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = FINAL_MANIFEST.with_name(
        f"{FINAL_MANIFEST.name}.{os.getpid()}.tmp"
    )
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(FINAL_MANIFEST)


def verify_final_manifest() -> None:
    current = _current_final_manifest_payload()
    rows = current["variants"]
    if set(rows) != set(VARIANTS):
        raise RuntimeError(
            f"Cannot finalize UMi checkpoint manifest: only {len(rows)}/7 "
            "variants are complete."
        )

    if not FINAL_MANIFEST.is_file():
        _write_final_manifest(current)
        print("[FINAL-GUARD] initialized immutable manifest:", FINAL_MANIFEST)
        return

    stored = load_json(FINAL_MANIFEST)
    if stored.get("config_sha256") != current["config_sha256"]:
        raise RuntimeError("Final checkpoint manifest config hash mismatch.")
    stored_rows = stored.get("variants", {})
    if set(stored_rows) != set(VARIANTS):
        raise RuntimeError(
            f"Stored final checkpoint manifest has {len(stored_rows)}/7 variants."
        )
    for variant in VARIANTS:
        if stored_rows[variant].get("sha256") != rows[variant]["sha256"]:
            raise RuntimeError(f"Final checkpoint hash mismatch: {variant}")
        if int(stored_rows[variant].get("size_bytes", -1)) != int(
            rows[variant]["size_bytes"]
        ):
            raise RuntimeError(f"Final checkpoint size mismatch: {variant}")
    print("[FINAL-GUARD] PASS: all 7 UMi-trained checkpoints match the immutable manifest.")


def _point_status(
    root: Path,
    *,
    variant: str,
    receiver: str,
    ebno: float,
) -> dict[str, Any]:
    rows: dict[int, dict[str, Any]] = {}
    if root.exists():
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
                if abs(float(row.get("ebno_db")) - ebno) > 1e-9:
                    continue
                rows[int(row.get("chunk_idx", -1))] = row
            except Exception:
                continue
    errors = sum(
        int(float(row.get("block_errors", 0) or 0))
        for row in rows.values()
    )
    batches = sum(
        int(float(row.get("num_batches_run", 0) or 0))
        for row in rows.values()
    )
    blocks = sum(
        int(float(row.get("num_blocks", 0) or 0))
        for row in rows.values()
    )
    done = (
        (batches >= MIN_BATCHES and errors >= TARGET_ERRORS)
        or batches >= MAX_BATCHES
    )
    return {
        "done": done,
        "errors": errors,
        "batches": batches,
        "blocks": blocks,
    }


def eval_status(domain: str) -> int:
    domain = domain.lower()
    root = UMI_EVAL_ROOT if domain == "umi" else CDLC_EVAL_ROOT
    all_complete = True
    print(f"UMi-trained UPAIR evaluated on {domain.upper()}, U=3")
    for variant in VARIANTS:
        cells = []
        for ebno in EBNOS:
            status = _point_status(
                root,
                variant=variant,
                receiver="upair5g_lmmse",
                ebno=ebno,
            )
            all_complete &= bool(status["done"])
            mark = "D" if status["done"] else "M"
            cells.append(
                f"{ebno:+g}:{mark}({status['errors']}/{status['batches']})"
            )
        print(f"{variant:28s} " + "  ".join(cells))
    print(f"UMI_TRAINED_{domain.upper()}_EVAL_COMPLETE={int(all_complete)}")
    return 0 if all_complete else 1


def pending_eval_indices(domain: str) -> str:
    root = UMI_EVAL_ROOT if domain.lower() == "umi" else CDLC_EVAL_ROOT
    pending = []
    for index, variant in enumerate(VARIANTS):
        complete = all(
            _point_status(
                root,
                variant=variant,
                receiver="upair5g_lmmse",
                ebno=ebno,
            )["done"]
            for ebno in EBNOS
        )
        if not complete:
            pending.append(str(index))
    return ",".join(pending)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("static")
    sub.add_parser("smoke-status")
    sub.add_parser("training-status")
    sub.add_parser("final-guard")

    p = sub.add_parser("variant")
    p.add_argument("index", type=int)

    p = sub.add_parser("prepare-study")
    p.add_argument("stage", choices=["A", "B", "a", "b"])
    p.add_argument("variant")

    p = sub.add_parser("study-status")
    p.add_argument("stage", choices=["A", "B", "a", "b"])

    p = sub.add_parser("pending-stage")
    p.add_argument("stage", choices=["A", "B", "a", "b"])

    p = sub.add_parser("smoke-train")
    p.add_argument("variant")

    p = sub.add_parser("train-final")
    p.add_argument("variant")

    p = sub.add_parser("eval-status")
    p.add_argument("domain", choices=["umi", "cdlc"])

    p = sub.add_parser("pending-eval")
    p.add_argument("domain", choices=["umi", "cdlc"])

    args = parser.parse_args()

    if args.command == "static":
        static_check()
    elif args.command == "variant":
        print(VARIANTS[args.index])
    elif args.command == "prepare-study":
        prepare_study(args.stage, args.variant)
    elif args.command == "study-status":
        raise SystemExit(print_study_status(args.stage))
    elif args.command == "pending-stage":
        print(pending_indices_for_stage(args.stage))
    elif args.command == "smoke-train":
        smoke_train(args.variant)
    elif args.command == "smoke-status":
        raise SystemExit(smoke_status())
    elif args.command == "train-final":
        train_final(args.variant)
    elif args.command == "training-status":
        raise SystemExit(final_training_status())
    elif args.command == "final-guard":
        verify_final_manifest()
        original_guard()
    elif args.command == "eval-status":
        raise SystemExit(eval_status(args.domain))
    elif args.command == "pending-eval":
        print(pending_eval_indices(args.domain))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
