#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from upair5g.config import set_cfg  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "umi_training_driver",
    ROOT / "umi_training" / "driver.py",
)
driver = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(driver)

_orig_apply_final_common = driver._apply_final_common


def _apply_final_common_lowmem(cfg, *, steps: int) -> None:
    _orig_apply_final_common(cfg, steps=steps)

    train_b = int(os.environ.get("UPAIR_FINAL_TRAIN_BATCH_SIZE", "16"))
    eval_b = int(os.environ.get("UPAIR_FINAL_EVAL_BATCH_SIZE", "16"))
    val_mb = int(os.environ.get("UPAIR_FINAL_VAL_MICROBATCH_SIZE", "8"))
    cleanup = int(os.environ.get("UPAIR_FINAL_MEMORY_CLEANUP_EVERY_STEPS", "25"))

    set_cfg(cfg, "system.batch_size_train", train_b)
    set_cfg(cfg, "system.batch_size_eval", eval_b)
    set_cfg(cfg, "training.val_microbatch_size", val_mb)
    set_cfg(cfg, "training.memory_cleanup_every_steps", cleanup)
    set_cfg(cfg, "training.val_memory_cleanup_every_microbatch", True)
    set_cfg(cfg, "training.memory_cleanup_after_validation", True)


driver._apply_final_common = _apply_final_common_lowmem


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: train_final_lowmem.py <variant>")
    driver.train_final(sys.argv[1])
