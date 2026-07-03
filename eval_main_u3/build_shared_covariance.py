#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.run_comprehensive_mu32_ablation import _apply_optuna_best_1dmrs, _eval_cfg, _variant_cfg
from upair5g.baselines import estimate_empirical_covariances
from upair5g.builders import build_channel, build_pusch_transmitter
from upair5g.config import ensure_output_tree, load_config, set_cfg
from upair5g.utils import set_global_seed

VARIANT = "main_d256_b4_r2"
PREFIX = "umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"
COV_SEED = 7007

cfg0 = load_config(ROOT / "configs" / "twc_comprehensive_mu32_umi_training.yaml")
train_cfg = _variant_cfg(cfg0, VARIANT, "1dmrs", 7)
_apply_optuna_best_1dmrs(
    train_cfg,
    VARIANT,
    "1dmrs",
    storage_dir=ROOT / "optuna",
    study_prefix=PREFIX,
    require_external=True,
)

cfg = _eval_cfg(train_cfg, VARIANT, "1dmrs", 3)
set_cfg(cfg, "system.seed", COV_SEED)
set_cfg(cfg, "system.evaluation_seed", COV_SEED)
set_cfg(cfg, "multiuser.fixed_num_users", 3)
set_cfg(cfg, "experiment.output_root", str(ROOT / "_main_umitrained_u3_shared_cov"))
set_cfg(cfg, "experiment.name", "u3_umi")
set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", True)
set_cfg(cfg, "baselines.covariance_estimation.cache_name", "empirical_covariances.npz")
set_cfg(cfg, "baselines.covariance_estimation.batch_size", 16)
set_cfg(cfg, "baselines.covariance_estimation.num_batches", 64)

set_global_seed(COV_SEED)
paths = ensure_output_tree(cfg)
cache = paths["artifacts"] / "empirical_covariances.npz"

if cache.exists():
    print("[COV] reuse:", cache)
    raise SystemExit(0)

tx, _ = build_pusch_transmitter(cfg, num_users=3)
channel = build_channel(cfg, tx)
result = estimate_empirical_covariances(tx=tx, channel=channel, cfg=cfg, paths=paths)

manifest = {
    "variant": VARIANT,
    "num_users": 3,
    "channel": "UMi normalized",
    "seed": COV_SEED,
    "cache": str(cache),
    "covariance_batches": 64,
    "covariance_batch_size": 16,
}
(paths["artifacts"] / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("[COV] wrote:", cache)
