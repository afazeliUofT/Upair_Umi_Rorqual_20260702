from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
SRC = ROOT / "_main_umitrained_u3_eval_chunks"
OUT = ROOT / "BLER_main_u3_eval-results"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT = "main_d256_b4_r2"
NUM_USERS = 3

LABELS = {
    "upair5g_lmmse": "UPAIR Main (d=256, L=4, r=2)",
    "baseline_ls_lmmse": "LS + LMMSE",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D-LMMSE + LMMSE",
    "perfect_csi_lmmse": "Perfect CSI + LMMSE",
}

ORDER = [
    "UPAIR Main (d=256, L=4, r=2)",
    "LS + LMMSE",
    "LS + 2D-LMMSE + LMMSE",
    "Perfect CSI + LMMSE",
]

MARKERS = {
    "UPAIR Main (d=256, L=4, r=2)": "o",
    "LS + LMMSE": "s",
    "LS + 2D-LMMSE + LMMSE": "^",
    "Perfect CSI + LMMSE": "D",
}

paths = sorted(SRC.rglob("chunk_result.csv"))
if not paths:
    raise FileNotFoundError(f"No chunk_result.csv files found under {SRC}")

frames = []
for p in paths:
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if df.empty:
        continue
    df["chunk_result_path"] = str(p)
    frames.append(df)

if not frames:
    raise RuntimeError("No readable chunk_result.csv files.")

raw = pd.concat(frames, ignore_index=True, sort=False)

required = ["variant", "receiver", "num_users", "ebno_db", "block_errors", "num_blocks", "num_batches_run"]
missing = [c for c in required if c not in raw.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

raw = raw[
    raw["variant"].astype(str).eq(VARIANT)
    & raw["receiver"].astype(str).isin(LABELS)
    & raw["num_users"].astype(int).eq(NUM_USERS)
].copy()

if raw.empty:
    raise RuntimeError("No U=3 main-variant rows found for UPAIR/baselines.")

for c in ["ebno_db", "block_errors", "num_blocks", "num_batches_run", "bit_errors", "num_bits"]:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

# Deduplicate by receiver/SNR/chunk index if chunk_idx exists.
if "chunk_idx" in raw.columns:
    raw["chunk_idx"] = pd.to_numeric(raw["chunk_idx"], errors="coerce").fillna(-1).astype(int)
    raw = raw.sort_values("chunk_result_path").drop_duplicates(
        ["variant", "receiver", "num_users", "ebno_db", "chunk_idx"],
        keep="last",
    )

group_cols = ["variant", "receiver", "num_users", "ebno_db"]

agg = (
    raw.groupby(group_cols, as_index=False)
    .agg(
        block_errors=("block_errors", "sum"),
        num_blocks=("num_blocks", "sum"),
        num_batches_run=("num_batches_run", "sum"),
        bit_errors=("bit_errors", "sum") if "bit_errors" in raw.columns else ("block_errors", "sum"),
        num_bits=("num_bits", "sum") if "num_bits" in raw.columns else ("num_blocks", "sum"),
    )
)

agg["bler"] = agg["block_errors"] / agg["num_blocks"].replace(0, np.nan)
agg["ber"] = agg["bit_errors"] / agg["num_bits"].replace(0, np.nan)
agg["reliable_bler"] = agg["block_errors"] >= 100
agg["plot_label"] = agg["receiver"].map(LABELS)

agg = agg.sort_values(["receiver", "ebno_db"])
agg.to_csv(OUT / "main_u3_all_aggregated_rows.csv", index=False)

# Plot only nonzero positive BLER points.
plot_df = agg[np.isfinite(agg["bler"]) & (agg["bler"] > 0)].copy()
plot_df.to_csv(OUT / "main_u3_bler_plot_data_nonzero.csv", index=False)

print("\n=== Aggregated rows ===")
print(
    agg[
        [
            "receiver",
            "plot_label",
            "ebno_db",
            "bler",
            "block_errors",
            "num_blocks",
            "num_batches_run",
            "reliable_bler",
        ]
    ].to_string(index=False)
)

print("\n=== Rows used in plot, BLER > 0 only ===")
print(
    plot_df[
        [
            "plot_label",
            "ebno_db",
            "bler",
            "block_errors",
            "num_blocks",
            "num_batches_run",
            "reliable_bler",
        ]
    ].to_string(index=False)
)

if plot_df.empty:
    raise RuntimeError("No positive BLER points to plot.")

plt.figure(figsize=(8.4, 6.2))

for label in ORDER:
    d = plot_df[plot_df["plot_label"].eq(label)].sort_values("ebno_db")
    if d.empty:
        print(f"[WARN] No plotted points for: {label}")
        continue

    plt.semilogy(
        d["ebno_db"],
        d["bler"],
        marker=MARKERS[label],
        linewidth=2.0,
        markersize=6,
        label=label,
    )

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.title("UMi, U=3: UPAIR Main vs Baselines")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

png = OUT / "bler_main_u3_vs_baselines.png"
pdf = OUT / "bler_main_u3_vs_baselines.pdf"
svg = OUT / "bler_main_u3_vs_baselines.svg"

plt.savefig(png, dpi=350, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(svg, bbox_inches="tight")
plt.close()

print("\n[OK] Wrote:")
print(" ", png)
print(" ", pdf)
print(" ", svg)
print(" ", OUT / "main_u3_all_aggregated_rows.csv")
print(" ", OUT / "main_u3_bler_plot_data_nonzero.csv")
