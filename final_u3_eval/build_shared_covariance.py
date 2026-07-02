#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.run_comprehensive_mu32_ablation import (  # noqa: E402
    _apply_optuna_best_1dmrs,
    _eval_cfg,
    _variant_cfg,
)
from upair5g.baselines import estimate_empirical_covariances  # noqa: E402
from upair5g.builders import build_channel, build_pusch_transmitter  # noqa: E402
from upair5g.config import ensure_output_tree, load_config, set_cfg  # noqa: E402
from upair5g.utils import set_global_seed  # noqa: E402

VARIANT = "main_d256_b4_r2"
PREFIX = "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
SEED = 7
COV_SEED = 7007

base = load_config(ROOT / "configs" / "twc_comprehensive_mu32_base.yaml")
train_cfg = _variant_cfg(base, VARIANT, "1dmrs", SEED)
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
set_cfg(cfg, "experiment.output_root", str(ROOT / "_final_u3_shared_cov"))
set_cfg(cfg, "experiment.name", "u3_prb8_cdlC_covariance")
set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", True)
set_cfg(cfg, "baselines.covariance_estimation.cache_name", "empirical_covariances.npz")

set_global_seed(COV_SEED)
paths = ensure_output_tree(cfg)
tx, _ = build_pusch_transmitter(cfg, num_users=3)
channel = build_channel(cfg, tx)
result = estimate_empirical_covariances(tx=tx, channel=channel, cfg=cfg, paths=paths)
cache = Path(str(result["cache_path"].numpy().decode() if hasattr(result["cache_path"], "numpy") else result["cache_path"]))

manifest = {
    "cache": str(cache),
    "num_users": 3,
    "n_size_grid": int(cfg["pusch"]["n_size_grid"]),
    "channel_model": str(cfg["channel"]["model"]),
    "delay_spread_s": float(cfg["channel"]["delay_spread_s"]),
    "min_speed_mps": float(cfg["channel"]["min_speed_mps"]),
    "max_speed_mps": float(cfg["channel"]["max_speed_mps"]),
    "num_rx_ant": int(cfg["channel"]["num_rx_ant"]),
    "num_batches": int(cfg["baselines"]["covariance_estimation"]["num_batches"]),
    "batch_size": int(cfg["baselines"]["covariance_estimation"]["batch_size"]),
    "seed": COV_SEED,
}
(cache.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(cache)
