#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def safe_tag(x):
    return str(x).replace("-", "m").replace("+", "p").replace(".", "p").replace(",", "_")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--dmrs-case", default="1dmrs")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--num-users", type=int, required=True)
ap.add_argument("--receiver", required=True)
ap.add_argument("--ebno-db", type=float, required=True)
ap.add_argument("--chunk-idx", type=int, required=True)
ap.add_argument("--chunk-batches", type=int, default=20)
ap.add_argument("--receiver-microbatch-size", type=int, default=4)
ap.add_argument("--stageb-prefix", required=True)
ap.add_argument("--optuna-dir", required=True)
ap.add_argument("--output-root", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--shared-cov-cache", required=True)
args = ap.parse_args()

if args.receiver != "baseline_ls_2dlmmse_lmmse":
    raise SystemExit("This wrapper is only for baseline_ls_2dlmmse_lmmse")

shared = Path(args.shared_cov_cache).resolve()
if not shared.is_file():
    raise FileNotFoundError(shared)

cfg = yaml.safe_load(Path(args.config).read_text())
cache_name = str(
    cfg.get("baselines", {})
       .get("covariance_estimation", {})
       .get("cache_name", "empirical_covariances.npz")
)

tag = (
    f"{args.variant}_u{args.num_users}_{args.receiver}_"
    f"ebno{safe_tag(args.ebno_db)}_chunk{args.chunk_idx:04d}_"
    f"m{args.receiver_microbatch_size}_b{args.chunk_batches}"
)

target = Path(args.output_root).resolve() / tag / "artifacts" / cache_name
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(shared, target)

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_isolated_eval_chunk.py"),
    "--config", args.config,
    "--variant", args.variant,
    "--dmrs-case", args.dmrs_case,
    "--seed", str(args.seed),
    "--num-users", str(args.num_users),
    "--receiver", args.receiver,
    "--ebno-db", str(args.ebno_db),
    "--chunk-idx", str(args.chunk_idx),
    "--chunk-batches", str(args.chunk_batches),
    "--receiver-microbatch-size", str(args.receiver_microbatch_size),
    "--stageb-prefix", args.stageb_prefix,
    "--optuna-dir", args.optuna_dir,
    "--output-root", args.output_root,
    "--checkpoint", args.checkpoint,
]

print("[SHARED-COV] staged", shared, "->", target)
subprocess.run(cmd, check=True, cwd=ROOT)
