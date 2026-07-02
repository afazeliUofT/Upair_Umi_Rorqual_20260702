#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "temporary plots" / "final_u3"
OUT.mkdir(parents=True, exist_ok=True)
UPAIR_ROOT = ROOT / "_isolated_eval_chunks"
BASE_ROOT = ROOT / "_final_u3_baseline_chunks"
EBNOS = {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0}
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
LABELS = {
    "main_d256_b4_r2": "UPAIR main (d=256,L=4,r=2)",
    "shallow_d256_b2_r2": "UPAIR shallow (d=256,L=2,r=2)",
    "deep_d256_b6_r2": "UPAIR deep (d=256,L=6,r=2)",
    "narrow_d192_b4_r2": "UPAIR narrow (d=192,L=4,r=2)",
    "wide_d320_b4_r2": "UPAIR wide (d=320,L=4,r=2)",
    "wide_deep_d320_b6_r2": "UPAIR wide-deep (d=320,L=6,r=2)",
    "mlpwide_d256_b4_r4": "UPAIR MLP-wide (d=256,L=4,r=4)",
    "baseline_ls_lmmse": "LS + LMMSE detector",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D-LMMSE + LMMSE detector",
    "perfect_csi_lmmse": "Perfect CSI + LMMSE detector",
}
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]


def load_rows(root: Path, receivers: set[str]) -> pd.DataFrame:
    frames = []
    for p in root.glob("merged_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        df["source"] = str(p.relative_to(ROOT))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ("num_users", "ebno_db", "bler", "block_errors", "num_blocks", "num_batches_run"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[
        df["receiver"].astype(str).isin(receivers)
        & df["num_users"].eq(3)
        & df["ebno_db"].isin(EBNOS)
    ].copy()
    df = df.sort_values("num_blocks").drop_duplicates(
        ["variant", "receiver", "num_users", "ebno_db"], keep="last"
    )
    return df


upair = load_rows(UPAIR_ROOT, {"upair5g_lmmse"})
upair = upair[upair["variant"].isin(VARIANTS)]
baselines = load_rows(BASE_ROOT, set(BASELINES))
baselines = baselines[baselines["variant"].eq("main_d256_b4_r2")]
all_rows = pd.concat([upair, baselines], ignore_index=True, sort=False)
all_rows["reliable_bler"] = all_rows["block_errors"].fillna(0).ge(100)
all_rows["done"] = (
    (all_rows["num_batches_run"].fillna(0).ge(20) & all_rows["block_errors"].fillna(0).ge(100))
    | all_rows["num_batches_run"].fillna(0).ge(2000)
)
all_rows.to_csv(OUT / "final_u3_bler_rows.csv", index=False)

expected = []
for v in VARIANTS:
    for e in sorted(EBNOS):
        expected.append((v, "upair5g_lmmse", e))
for r in BASELINES:
    for e in sorted(EBNOS):
        expected.append(("main_d256_b4_r2", r, e))
observed = set(zip(all_rows["variant"], all_rows["receiver"], all_rows["ebno_db"]))
missing = pd.DataFrame(
    [(v, r, e) for v, r, e in expected if (v, r, e) not in observed],
    columns=["variant", "receiver", "ebno_db"],
)
missing.to_csv(OUT / "missing_points.csv", index=False)
all_rows[all_rows["bler"].fillna(0).le(0)].to_csv(OUT / "zero_bler_points.csv", index=False)


def plot(series_keys: list[tuple[str, str]], filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    for i, (variant, receiver) in enumerate(series_keys):
        s = all_rows[
            all_rows["variant"].eq(variant) & all_rows["receiver"].eq(receiver)
        ].sort_values("ebno_db")
        s = s[np.isfinite(s["bler"]) & s["bler"].gt(0)]
        if s.empty:
            continue
        key = variant if receiver == "upair5g_lmmse" else receiver
        linestyle = "-" if receiver == "upair5g_lmmse" else "--"
        line, = ax.plot(
            s["ebno_db"], s["bler"], linestyle=linestyle, linewidth=1.8,
            label=LABELS[key], zorder=2,
        )
        reliable = s["reliable_bler"].to_numpy(dtype=bool)
        x = s["ebno_db"].to_numpy(dtype=float)
        y = s["bler"].to_numpy(dtype=float)
        marker = MARKERS[i % len(MARKERS)]
        if reliable.any():
            ax.scatter(x[reliable], y[reliable], marker=marker, s=52,
                       color=line.get_color(), zorder=3)
        if (~reliable).any():
            ax.scatter(x[~reliable], y[~reliable], marker=marker, s=58,
                       facecolors="none", edgecolors=line.get_color(),
                       linewidths=1.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(sorted(EBNOS))
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("BLER")
    ax.set_title(title + "\n3 active users")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.text(
        0.01, 0.01,
        "Filled: ≥100 block errors; open: <100 errors at the 2000-batch cap. "
        "Zero/unavailable points are omitted.",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.2)
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)

upair_keys = [(v, "upair5g_lmmse") for v in VARIANTS]
base_keys = [("main_d256_b4_r2", r) for r in BASELINES]
plot(upair_keys + base_keys, "bler_all_7_upair_and_3_benchmarks_u3",
     "Extended UPAIR variants and benchmarks")
plot(upair_keys, "bler_all_7_upair_variants_u3", "Extended UPAIR architecture comparison")
plot([("main_d256_b4_r2", "upair5g_lmmse")] + base_keys,
     "bler_main_upair_vs_benchmarks_u3", "Main Extended UPAIR versus benchmarks")

print("[PLOT] rows:", len(all_rows))
print("[PLOT] missing points:", len(missing))
print("[PLOT] output:", OUT)
