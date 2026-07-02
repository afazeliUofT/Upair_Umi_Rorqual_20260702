#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
import yaml
import math

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
sys = cfg["system"]
tr = cfg["training"]
ev = cfg["evaluation"]

train_b = int(sys["batch_size_train"])
val_b = int(sys["batch_size_eval"])
val_mb = int(tr["val_microbatch_size"])
eval_logical_b = int(ev.get("logical_batch_size", val_b))
eval_mb = int(ev["receiver_microbatch_size"])
target_be = int(ev["target_block_errors_per_receiver"])
max_batches = int(ev["max_num_batches_per_point"])
min_batches = int(ev["min_num_batches_per_point"])
chunk_batches = int(float(__import__("os").environ.get("UPAIR_PIPELINE_CHUNK_BATCHES", "20")))

print("="*90)
print("[BATCH POLICY]")
print(f"training batch size                  = {train_b}")
print(f"Optuna/final validation batch size    = {val_b}")
print(f"validation microbatch size            = {val_mb}")
print(f"final BLER logical batch size         = {eval_logical_b}")
print(f"receiver microbatch size              = {eval_mb}")
print(f"isolated eval chunk_batches default   = {chunk_batches}")
print(f"min/max batches per point             = {min_batches}/{max_batches}")
print(f"target block errors per receiver      = {target_be}")
print()

if train_b == 32 and val_b == 32 and val_mb == 16:
    print("[OK] Training/validation batch policy matches Optuna defaults: train=32, val=32, val_micro=16.")
else:
    print("[WARN] Training/validation batch policy differs from Optuna defaults.")

if eval_logical_b >= eval_mb and eval_logical_b % eval_mb == 0:
    print("[OK] Evaluation logical batch is compatible with receiver microbatching.")
else:
    print("[WARN] Evaluation logical batch is not an integer multiple of receiver microbatch.")

print()
print("="*90)
print("[BLER RELIABILITY CALCULATIONS]")
for b in [32, 64, eval_logical_b]:
    for mb in [1500, 2000, 8000, 16000]:
        frames = b * mb
        exp_at_1e4 = frames * 1e-4
        ub_zero = 3.0 / frames
        print(
            f"batch={b:>3d}, max_batches={mb:>5d}: "
            f"frames={frames:>8d}, expected errors at 1e-4={exp_at_1e4:>6.1f}, "
            f"zero-error UB≈{ub_zero:.2e}"
        )

print()
for target in [50, 100]:
    frames_needed = int(math.ceil(target / 1e-4))
    print(f"To get {target} block errors at BLER=1e-4:")
    print(f"  frames needed = {frames_needed}")
    print(f"  batches at batch=32 = {math.ceil(frames_needed/32)}")
    print(f"  batches at batch=64 = {math.ceil(frames_needed/64)}")
PY
