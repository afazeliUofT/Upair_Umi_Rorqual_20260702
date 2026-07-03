from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "BLER_main_u3_eval-results"
OUT.mkdir(parents=True, exist_ok=True)

# Prefer the standard merged file; otherwise search for a reasonable CSV.
candidates = [
    ROOT / "_main_umitrained_u3_eval_chunks" / "main_u3_all_receivers_merged.csv",
    ROOT / "_main_umitrained_u3_eval_chunks" / "main_u3_all_receivers.csv",
]

csv_path = None
for p in candidates:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    found = sorted(ROOT.rglob("*main*u3*receivers*.csv"))
    if not found:
        found = sorted(ROOT.rglob("*all_receivers*merged*.csv"))
    if not found:
        raise FileNotFoundError("Could not find merged eval CSV.")
    csv_path = found[0]

print(f"[INFO] Using CSV: {csv_path}")

df = pd.read_csv(csv_path)

# Flexible column detection
cols = {c.lower(): c for c in df.columns}

def pick(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

receiver_col = pick("receiver", "receiver_name", "method", "scheme", "model")
ebno_col     = pick("ebno_db", "eb_n0_db", "ebn0_db", "snr_db")
bler_col     = pick("bler", "bler_mean")
u_col        = pick("num_users", "active_users", "u", "users")

if receiver_col is None or ebno_col is None or bler_col is None:
    raise RuntimeError(
        f"Missing required columns. Found columns: {list(df.columns)}"
    )

if u_col is not None:
    df = df[df[u_col] == 3].copy()

def normalize_receiver(x):
    s = str(x).strip().lower()

    if ("main_d256_b4_r2" in s) or ("upair main" in s) or (s == "main"):
        return "UPAIR Main (d=256, L=4, r=2)"

    if ("baseline_ls_2dlmmse_lmmse" in s) or ("2dlmmse" in s):
        return "LS + 2D-LMMSE + LMMSE"

    if ("baseline_ls_lmmse" in s) or (s == "ls_lmmse"):
        return "LS + LMMSE"

    if ("perfect_csi_lmmse" in s) or ("perfect csi" in s):
        return "Perfect CSI + LMMSE"

    return None

df["plot_label"] = df[receiver_col].map(normalize_receiver)
df["ebno_db_plot"] = pd.to_numeric(df[ebno_col], errors="coerce")
df["bler_plot"] = pd.to_numeric(df[bler_col], errors="coerce")

plot_df = df[
    df["plot_label"].notna()
    & np.isfinite(df["ebno_db_plot"])
    & np.isfinite(df["bler_plot"])
    & (df["bler_plot"] > 0)
].copy()

if plot_df.empty:
    raise RuntimeError("No valid nonzero BLER points found for the requested methods.")

plot_df = (
    plot_df.groupby(["plot_label", "ebno_db_plot"], as_index=False)["bler_plot"]
    .mean()
    .sort_values(["plot_label", "ebno_db_plot"])
)

plot_df.to_csv(OUT / "main_u3_bler_plot_data.csv", index=False)

order = [
    "UPAIR Main (d=256, L=4, r=2)",
    "LS + LMMSE",
    "LS + 2D-LMMSE + LMMSE",
    "Perfect CSI + LMMSE",
]
markers = {
    "UPAIR Main (d=256, L=4, r=2)": "o",
    "LS + LMMSE": "s",
    "LS + 2D-LMMSE + LMMSE": "^",
    "Perfect CSI + LMMSE": "D",
}

plt.figure(figsize=(8, 6))
for label in order:
    d = plot_df[plot_df["plot_label"] == label].sort_values("ebno_db_plot")
    if d.empty:
        continue
    plt.semilogy(
        d["ebno_db_plot"],
        d["bler_plot"],
        marker=markers[label],
        linewidth=2,
        markersize=6,
        label=label,
    )

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.title("UMi, U=3: UPAIR Main vs Baselines")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

png_path = OUT / "bler_main_u3_vs_baselines.png"
pdf_path = OUT / "bler_main_u3_vs_baselines.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("[OK] Wrote:")
print(" ", png_path)
print(" ", pdf_path)
print(" ", OUT / "main_u3_bler_plot_data.csv")
