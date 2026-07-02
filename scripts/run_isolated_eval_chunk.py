from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402
from scripts.run_comprehensive_mu32_ablation import (  # noqa: E402
    _apply_optuna_best_1dmrs,
    _eval_cfg,
    _variant_cfg,
)


def _safe_tag(x: Any) -> str:
    s = str(x)
    return s.replace("-", "m").replace("+", "p").replace(".", "p").replace(",", "_")


def _read_single_row(curves_path: Path, receiver: str, ebno: float, num_users: int) -> dict[str, Any]:
    df = pd.read_csv(curves_path)
    if "receiver" in df.columns:
        df = df[df["receiver"].astype(str) == str(receiver)]
    if "ebno_db" in df.columns:
        df = df[df["ebno_db"].astype(float) == float(ebno)]
    if "num_users" in df.columns:
        df = df[df["num_users"].astype(int) == int(num_users)]
    if len(df) != 1:
        raise RuntimeError(
            f"Expected one curve row for receiver={receiver}, ebno={ebno}, num_users={num_users}, "
            f"got {len(df)} rows from {curves_path}"
        )
    return df.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated evaluation chunk.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dmrs-case", default="1dmrs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--ebno-db", type=float, required=True)
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--chunk-batches", type=int, default=20)
    parser.add_argument("--receiver-microbatch-size", type=int, default=8)
    parser.add_argument("--stageb-prefix", default="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB")
    parser.add_argument("--optuna-dir", default=str(PROJECT_ROOT / "optuna"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "_isolated_eval_chunks"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_CPP_VMODULE", "bfc_allocator=0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    cfg = load_config(args.config)
    train_cfg = _variant_cfg(cfg, args.variant, args.dmrs_case, args.seed)
    _apply_optuna_best_1dmrs(
        train_cfg,
        args.variant,
        args.dmrs_case,
        storage_dir=args.optuna_dir,
        study_prefix=args.stageb_prefix,
        require_external=True,
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        PROJECT_ROOT
        / "TWC_plots_comprehensive"
        / "runs_rx16"
        / f"seed{args.seed}"
        / args.dmrs_case
        / args.variant
        / "checkpoints"
        / str(get_cfg(train_cfg, "training.checkpoint_name", "best.weights.h5"))
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for isolated eval: {checkpoint}")

    cfg_eval = _eval_cfg(train_cfg, args.variant, args.dmrs_case, args.num_users)

    # Use a distinct deterministic evaluation seed per chunk so chunks are independent
    # rather than replaying the same Monte Carlo samples.
    base_eval_seed = int(get_cfg(cfg_eval, "system.evaluation_seed", args.seed + 1000))
    ebno_offset = int(round((float(args.ebno_db) + 100.0) * 1000.0))
    chunk_seed = base_eval_seed + 100003 * int(args.chunk_idx) + 17 * ebno_offset + 1009 * int(args.num_users)
    set_cfg(cfg_eval, "system.evaluation_seed", int(chunk_seed))
    set_cfg(cfg_eval, "system.seed", int(chunk_seed))

    tag = (
        f"{args.variant}_u{args.num_users}_{args.receiver}_"
        f"ebno{_safe_tag(args.ebno_db)}_chunk{args.chunk_idx:04d}_"
        f"m{args.receiver_microbatch_size}_b{args.chunk_batches}"
    )
    out_root = Path(args.output_root)
    set_cfg(cfg_eval, "experiment.output_root", str(out_root))
    set_cfg(cfg_eval, "experiment.name", tag)

    set_cfg(cfg_eval, "system.ebno_db_eval", [float(args.ebno_db)])
    set_cfg(cfg_eval, "baselines.enabled_receivers", [str(args.receiver)])

    # BLER-only, memory-safe, no long-lived compiled receiver-count graph.
    set_cfg(cfg_eval, "evaluation.nmse_receivers", [])
    set_cfg(cfg_eval, "evaluation.save_example_batch", False)
    set_cfg(cfg_eval, "evaluation.compiled_receiver_error_counts", False)
    set_cfg(cfg_eval, "evaluation.receiver_call_jit_compile", False)
    set_cfg(cfg_eval, "evaluation.receiver_microbatch_size", int(args.receiver_microbatch_size))
    set_cfg(cfg_eval, "evaluation.stream_eval_microbatches", True)
    set_cfg(cfg_eval, "evaluation.memory_cleanup_every_batches", 1)
    set_cfg(cfg_eval, "evaluation.memory_cleanup_every_microbatch", True)
    set_cfg(cfg_eval, "evaluation.min_num_batches_per_point", int(args.chunk_batches))
    set_cfg(cfg_eval, "evaluation.max_num_batches_per_point", int(args.chunk_batches))
    set_cfg(cfg_eval, "evaluation.target_block_errors_per_receiver", 0)
    set_cfg(cfg_eval, "evaluation.per_receiver_stopping", False)
    set_cfg(cfg_eval, "evaluation.force", True)
    set_cfg(cfg_eval, "evaluation.progress_every_batches", max(1, min(10, int(args.chunk_batches))))
    set_cfg(cfg_eval, "baselines.covariance_estimation.reuse_cache", True)

    print("[ISO-CHUNK] variant:", args.variant)
    print("[ISO-CHUNK] receiver:", args.receiver)
    print("[ISO-CHUNK] num_users:", args.num_users)
    print("[ISO-CHUNK] ebno_db:", args.ebno_db)
    print("[ISO-CHUNK] chunk_idx:", args.chunk_idx)
    print("[ISO-CHUNK] chunk_batches:", args.chunk_batches)
    print("[ISO-CHUNK] receiver_microbatch_size:", args.receiver_microbatch_size)
    print("[ISO-CHUNK] chunk_seed:", chunk_seed)
    print("[ISO-CHUNK] checkpoint:", checkpoint)
    print("[ISO-CHUNK] TF_GPU_ALLOCATOR:", os.environ.get("TF_GPU_ALLOCATOR"))
    print("[ISO-CHUNK] output tag:", tag)

    result = evaluate_model(cfg_eval, checkpoint_path=str(checkpoint), num_users=int(args.num_users))
    curves_path = Path(result["curves_path"])
    row = _read_single_row(curves_path, args.receiver, float(args.ebno_db), int(args.num_users))
    row.update(
        {
            "variant": args.variant,
            "dmrs_case": args.dmrs_case,
            "seed": int(args.seed),
            "training_seed": int(args.seed),
            "evaluation_seed": int(chunk_seed),
            "receiver": args.receiver,
            "num_users": int(args.num_users),
            "ebno_db": float(args.ebno_db),
            "chunk_idx": int(args.chunk_idx),
            "chunk_batches_requested": int(args.chunk_batches),
            "receiver_microbatch_size": int(args.receiver_microbatch_size),
            "checkpoint_path": str(checkpoint),
            "chunk_output_dir": str(result["output_dir"]),
            "chunk_curves_path": str(curves_path),
        }
    )

    out_dir = Path(result["output_dir"])
    pd.DataFrame([row]).to_csv(out_dir / "chunk_result.csv", index=False)
    with open(out_dir / "chunk_result.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)
    print("[ISO-CHUNK] wrote:", out_dir / "chunk_result.csv")
    print("[ISO-CHUNK] DONE")


if __name__ == "__main__":
    main()
