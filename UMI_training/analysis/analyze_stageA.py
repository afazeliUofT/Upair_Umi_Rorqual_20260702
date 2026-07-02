from pathlib import Path
import json
import math

import optuna
import pandas as pd

ROOT = Path.cwd()

VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]

PREFIX = "umiNorm_v1_b32_prb8_u34610_1dmrs_stageA"
DB_ROOT = ROOT / "UMI_training" / "optuna_db"
RUNS_ROOT = ROOT / "optuna" / "runs_1dmrs"
OUT = ROOT / "UMI_training" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def read_history(study_name, variant, trial_number):
    path = RUNS_ROOT / study_name / f"{variant}_trial_{trial_number:04d}" / "metrics" / "history.json"
    if not path.exists():
        return {}

    try:
        rows = json.loads(path.read_text()).get("history", [])
    except Exception:
        return {}

    val_rows = [
        r for r in rows
        if isinstance(r, dict) and "val_nmse_prop" in r
    ]

    if not val_rows:
        return {}

    last = val_rows[-1]
    best = min(val_rows, key=lambda r: float(r["val_nmse_prop"]))

    return {
        "last_step": int(last.get("step", -1)),
        "last_val_nmse_prop": float(last["val_nmse_prop"]),
        "last_val_nmse_ls": float(last.get("val_nmse_ls", math.nan)),
        "last_val_nmse_ratio": float(last["val_nmse_prop"]) / max(float(last.get("val_nmse_ls", math.nan)), 1e-12),
        "best_val_nmse_prop": float(best["val_nmse_prop"]),
        "best_val_nmse_ls": float(best.get("val_nmse_ls", math.nan)),
        "best_val_nmse_ratio": float(best["val_nmse_prop"]) / max(float(best.get("val_nmse_ls", math.nan)), 1e-12),
        "best_nmse_step": int(best.get("step", -1)),
    }


all_rows = []
summary_rows = []

for variant in VARIANTS:
    study_name = f"{PREFIX}_{variant}"
    db = DB_ROOT / f"{study_name}.db"
    storage = f"sqlite:///{db.resolve()}"

    if not db.exists():
        raise FileNotFoundError(db)

    study = optuna.load_study(study_name=study_name, storage=storage)

    rows = []
    for t in study.trials:
        attrs = dict(t.user_attrs)

        # The workflow queued the CDL-C Stage-B best as the first Stage-A candidate.
        is_warm = (
            attrs.get("source_domain") == "cdl_c"
            or attrs.get("purpose") == "UMi Stage-A warm start"
            or int(t.number) == 0
        )

        h = read_history(study_name, variant, int(t.number))

        row = {
            "variant": variant,
            "trial": int(t.number),
            "state": t.state.name,
            "value": float(t.value) if t.value is not None else math.nan,
            "is_warm_start": bool(is_warm),
            "source_domain": attrs.get("source_domain", ""),
            "purpose": attrs.get("purpose", ""),
            **h,
        }

        for k, v in t.params.items():
            row[f"param_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    all_rows.append(df)

    complete = df[df["state"].eq("COMPLETE") & df["value"].notna()].copy()
    complete = complete.sort_values("value", ascending=True)

    warm = df[df["is_warm_start"]].sort_values("trial").head(1)
    if warm.empty:
        warm = df[df["trial"].eq(0)].copy()

    best = complete.head(1)

    if best.empty:
        raise RuntimeError(f"No completed trials for {variant}")

    w = warm.iloc[0]
    b = best.iloc[0]

    warm_value = float(w["value"]) if pd.notna(w["value"]) else math.nan
    best_value = float(b["value"])

    if math.isfinite(warm_value):
        delta_log = warm_value - best_value
        # Objective is log-scale. This is improvement in the objective's underlying scale,
        # not necessarily pure NMSE improvement.
        objective_factor = 10.0 ** delta_log
        percent_lower = 100.0 * (1.0 - 1.0 / objective_factor) if objective_factor > 0 else math.nan
        improved = best_value < warm_value
    else:
        delta_log = math.nan
        objective_factor = math.nan
        percent_lower = math.nan
        improved = False

    rank_of_warm = None
    if int(w["trial"]) in set(complete["trial"].astype(int)):
        ranked = complete.reset_index(drop=True)
        pos = ranked.index[ranked["trial"].astype(int).eq(int(w["trial"]))]
        rank_of_warm = int(pos[0]) + 1 if len(pos) else None

    summary_rows.append({
        "variant": variant,
        "warm_trial": int(w["trial"]),
        "warm_state": str(w["state"]),
        "warm_value": warm_value,
        "warm_rank_among_completed": rank_of_warm,
        "best_trial": int(b["trial"]),
        "best_value": best_value,
        "best_state": str(b["state"]),
        "delta_log_warm_minus_best": delta_log,
        "objective_factor_best_vs_warm": objective_factor,
        "objective_percent_lower_than_warm": percent_lower,
        "stageA_improved_over_warm": bool(improved),
        "completed_trials": int(len(complete)),
        "pruned_trials": int(df["state"].eq("PRUNED").sum()),
        "failed_trials": int(df["state"].eq("FAIL").sum()),
    })

    top6 = complete.head(6).copy()
    print("\n" + "=" * 100)
    print(f"{variant}")
    print("=" * 100)
    print("Warm-start trial:")
    print(
        warm[[
            "trial", "state", "value",
            "last_step", "last_val_nmse_prop", "last_val_nmse_ls", "last_val_nmse_ratio",
            "param_learning_rate_schedule", "param_learning_rate",
            "param_weight_decay", "param_nmse_loss_weight",
            "param_dropout", "param_residual_scale",
        ]].to_string(index=False)
    )

    print("\nTop 6 completed Stage-A trials; these are the candidates Stage B will promote:")
    show_cols = [
        "trial", "state", "value",
        "last_step", "last_val_nmse_prop", "last_val_nmse_ls", "last_val_nmse_ratio",
        "param_learning_rate_schedule", "param_learning_rate",
        "param_weight_decay", "param_nmse_loss_weight",
        "param_grad_clip_norm", "param_dropout", "param_residual_scale",
    ]
    show_cols = [c for c in show_cols if c in top6.columns]
    print(top6[show_cols].to_string(index=False))

    print("\nImprovement relative to warm-start:")
    if math.isfinite(warm_value):
        print(f"  warm value = {warm_value:.6g}")
        print(f"  best value = {best_value:.6g}")
        print(f"  delta_log  = {delta_log:.6g}  positive means improvement")
        print(f"  objective-scale factor = {objective_factor:.3f}x")
        print(f"  objective-scale reduction = {percent_lower:.1f}%")
    else:
        print("  warm-start trial has no final value; it was likely pruned/unfinished.")


all_df = pd.concat(all_rows, ignore_index=True)
summary = pd.DataFrame(summary_rows)

all_df.to_csv(OUT / "stageA_all_trials.csv", index=False)
summary.to_csv(OUT / "stageA_warm_vs_best_summary.csv", index=False)

print("\n" + "#" * 100)
print("SUMMARY: Stage A warm-start versus best completed trial")
print("#" * 100)
print(summary.to_string(index=False))

print("\nWrote:")
print(" ", OUT / "stageA_all_trials.csv")
print(" ", OUT / "stageA_warm_vs_best_summary.csv")
