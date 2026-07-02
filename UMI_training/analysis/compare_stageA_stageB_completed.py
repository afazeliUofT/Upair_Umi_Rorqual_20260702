from pathlib import Path
import json
import math

import optuna
import pandas as pd
from optuna.trial import TrialState

ROOT = Path.cwd()
OUT = ROOT / "UMI_training" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
]

PREFIX_A = "umiNorm_v1_b32_prb8_u34610_1dmrs_stageA"
PREFIX_B = "umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"

DB_ROOT = ROOT / "UMI_training" / "optuna_db"
RUN_ROOT = ROOT / "optuna" / "runs_1dmrs"


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def study_name(stage_prefix, variant):
    return f"{stage_prefix}_{variant}"


def trial_dir(stage_prefix, variant, trial_number):
    name = study_name(stage_prefix, variant)
    return RUN_ROOT / name / f"{variant}_trial_{int(trial_number):04d}"


def read_trial_history(stage_prefix, variant, trial_number):
    path = trial_dir(stage_prefix, variant, trial_number) / "metrics" / "history.json"
    payload = load_json(path)
    rows = [
        r for r in payload.get("history", [])
        if isinstance(r, dict) and "val_nmse_prop" in r
    ]

    if not rows:
        return {
            "last_val_step": math.nan,
            "last_val_nmse_prop": math.nan,
            "last_val_nmse_ls": math.nan,
            "last_val_nmse_ratio": math.nan,
            "best_val_nmse_prop": math.nan,
            "best_nmse_step": math.nan,
        }

    last = rows[-1]
    best = min(rows, key=lambda r: float(r["val_nmse_prop"]))

    last_prop = float(last["val_nmse_prop"])
    last_ls = float(last.get("val_nmse_ls", math.nan))
    best_prop = float(best["val_nmse_prop"])

    return {
        "last_val_step": int(last.get("step", -1)),
        "last_val_nmse_prop": last_prop,
        "last_val_nmse_ls": last_ls,
        "last_val_nmse_ratio": last_prop / max(last_ls, 1e-12),
        "best_val_nmse_prop": best_prop,
        "best_nmse_step": int(best.get("step", -1)),
    }


def read_trial_config(stage_prefix, variant, trial_number):
    path = trial_dir(stage_prefix, variant, trial_number) / "artifacts" / "trial_config.json"
    cfg = load_json(path)
    return {
        "batch_size_train": cfg.get("system", {}).get("batch_size_train", ""),
        "batch_size_eval": cfg.get("system", {}).get("batch_size_eval", ""),
        "val_microbatch_size": cfg.get("training", {}).get("val_microbatch_size", ""),
        "memory_cleanup_every_steps": cfg.get("training", {}).get("memory_cleanup_every_steps", ""),
        "steps": cfg.get("training", {}).get("steps", ""),
    }


def param_equal(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-10 * max(1.0, abs(float(a)), abs(float(b)))
    return str(a) == str(b)


def same_params(pa, pb):
    keys = set(pa) | set(pb)
    for k in keys:
        if k not in pa or k not in pb:
            return False
        if not param_equal(pa[k], pb[k]):
            return False
    return True


def load_trials(stage_prefix, variant):
    name = study_name(stage_prefix, variant)
    db = DB_ROOT / f"{name}.db"
    if not db.exists():
        raise FileNotFoundError(db)

    study = optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{db.resolve()}",
    )

    rows = []
    for t in study.get_trials(deepcopy=True):
        h = read_trial_history(stage_prefix, variant, int(t.number))
        c = read_trial_config(stage_prefix, variant, int(t.number))

        row = {
            "variant": variant,
            "stage": "A" if stage_prefix == PREFIX_A else "B",
            "study_name": name,
            "trial": int(t.number),
            "state": t.state.name,
            "value": float(t.value) if t.value is not None else math.nan,
            **h,
            **c,
        }

        for k, v in t.params.items():
            row[f"param_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def params_from_row(row):
    params = {}
    for k, v in row.items():
        if not str(k).startswith("param_"):
            continue
        if pd.isna(v):
            continue
        params[str(k).replace("param_", "", 1)] = v
    return params


all_tables = []
summary_rows = []

for variant in VARIANTS:
    A = load_trials(PREFIX_A, variant)
    B = load_trials(PREFIX_B, variant)

    all_tables.extend([A, B])

    A_complete = A[A["state"].eq("COMPLETE") & A["value"].notna()].copy()
    B_complete = B[B["state"].eq("COMPLETE") & B["value"].notna()].copy()

    A_complete = A_complete.sort_values("value", ascending=True).reset_index(drop=True)
    B_complete = B_complete.sort_values("value", ascending=True).reset_index(drop=True)

    A_complete["stageA_rank"] = range(1, len(A_complete) + 1)

    print("\n" + "=" * 110)
    print(variant)
    print("=" * 110)

    if A_complete.empty:
        print("[WARN] No completed Stage-A trials")
        continue

    if B_complete.empty:
        print("[WARN] No completed Stage-B trials")
        continue

    best_A = A_complete.iloc[0]
    best_B = B_complete.iloc[0]

    # Match Stage-B best to the Stage-A candidate from which it came.
    matched_A = None
    b_params = params_from_row(best_B)

    for _, a_row in A_complete.iterrows():
        a_params = params_from_row(a_row)
        if same_params(a_params, b_params):
            matched_A = a_row
            break

    A_best_value = float(best_A["value"])
    B_best_value = float(best_B["value"])
    delta_global = A_best_value - B_best_value
    factor_global = 10.0 ** delta_global

    if matched_A is not None:
        matched_A_value = float(matched_A["value"])
        delta_same_candidate = matched_A_value - B_best_value
        factor_same_candidate = 10.0 ** delta_same_candidate
        matched_A_trial = int(matched_A["trial"])
        matched_A_rank = int(matched_A["stageA_rank"])
    else:
        matched_A_value = math.nan
        delta_same_candidate = math.nan
        factor_same_candidate = math.nan
        matched_A_trial = None
        matched_A_rank = None

    summary_rows.append({
        "variant": variant,
        "stageA_completed": int(len(A_complete)),
        "stageB_completed": int(len(B_complete)),
        "stageA_best_trial": int(best_A["trial"]),
        "stageA_best_value": A_best_value,
        "stageB_best_trial": int(best_B["trial"]),
        "stageB_best_value": B_best_value,
        "stageB_best_matched_stageA_trial": matched_A_trial,
        "stageB_best_matched_stageA_rank": matched_A_rank,
        "stageB_best_matched_stageA_value": matched_A_value,
        "stageB_minus_stageA_global_delta_log": delta_global,
        "stageB_global_objective_factor": factor_global,
        "stageB_global_percent_lower": 100.0 * (1.0 - 1.0 / factor_global),
        "stageB_same_candidate_delta_log": delta_same_candidate,
        "stageB_same_candidate_factor": factor_same_candidate,
        "stageB_same_candidate_percent_lower": (
            100.0 * (1.0 - 1.0 / factor_same_candidate)
            if math.isfinite(factor_same_candidate) and factor_same_candidate > 0
            else math.nan
        ),
        "stageB_helped_vs_stageA_best": bool(B_best_value < A_best_value),
        "stageB_helped_same_candidate": (
            bool(B_best_value < matched_A_value)
            if math.isfinite(matched_A_value)
            else None
        ),
        "stageB_batch_size_train": best_B.get("batch_size_train", ""),
        "stageB_batch_size_eval": best_B.get("batch_size_eval", ""),
        "stageB_val_microbatch_size": best_B.get("val_microbatch_size", ""),
    })

    print("\nStage-A best:")
    print(
        best_A[[
            "trial", "value", "last_val_step",
            "last_val_nmse_prop", "last_val_nmse_ls", "last_val_nmse_ratio",
            "batch_size_train", "batch_size_eval", "val_microbatch_size",
        ]].to_string()
    )

    print("\nStage-B best:")
    print(
        best_B[[
            "trial", "value", "last_val_step",
            "last_val_nmse_prop", "last_val_nmse_ls", "last_val_nmse_ratio",
            "batch_size_train", "batch_size_eval", "val_microbatch_size",
        ]].to_string()
    )

    print("\nComparison:")
    print(f"  Stage-B best value                         = {B_best_value:.6g}")
    print(f"  Stage-A global best value                  = {A_best_value:.6g}")
    print(f"  Improvement over Stage-A global best       = {delta_global:.6g} log10 units")
    print(f"  Objective-scale factor                     = {factor_global:.3f}x")
    print(f"  Objective-scale reduction                  = {100.0 * (1.0 - 1.0 / factor_global):.2f}%")

    if matched_A is not None:
        print(f"  Stage-B best came from Stage-A trial       = {matched_A_trial}")
        print(f"  That candidate's Stage-A rank              = {matched_A_rank}")
        print(f"  That candidate's Stage-A value             = {matched_A_value:.6g}")
        print(f"  Improvement of same candidate in Stage-B   = {delta_same_candidate:.6g} log10 units")
        print(f"  Same-candidate factor                      = {factor_same_candidate:.3f}x")
    else:
        print("  Could not match Stage-B best params to a Stage-A trial.")

    print("\nTop Stage-B completed trials:")
    show_cols = [
        "trial", "value", "last_val_step",
        "last_val_nmse_prop", "last_val_nmse_ratio",
        "param_learning_rate_schedule",
        "param_learning_rate",
        "param_weight_decay",
        "param_nmse_loss_weight",
        "param_dropout",
        "param_residual_scale",
    ]
    show_cols = [c for c in show_cols if c in B_complete.columns]
    print(B_complete[show_cols].head(6).to_string(index=False))


all_df = pd.concat(all_tables, ignore_index=True)
summary = pd.DataFrame(summary_rows)

all_df.to_csv(OUT / "stageA_stageB_all_trials_completed_variants.csv", index=False)
summary.to_csv(OUT / "stageA_vs_stageB_summary_completed_variants.csv", index=False)

print("\n" + "#" * 110)
print("SUMMARY")
print("#" * 110)
print(summary.to_string(index=False))

print("\nWrote:")
print(" ", OUT / "stageA_stageB_all_trials_completed_variants.csv")
print(" ", OUT / "stageA_vs_stageB_summary_completed_variants.csv")
