#!/usr/bin/env python3
"""Create comprehensive UPAIR 3-user BLER, timing, and training plots.

Run from the repository root:
    python make_BLER_3u_eval_results_comprehensive.py

Or provide the repository path explicitly:
    python make_BLER_3u_eval_results_comprehensive.py /path/to/repository
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


U = 3
EBNOS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
TARGET_BLOCK_ERRORS = 100
MAX_BATCHES = 2000
MIN_BATCHES = 20
TRADEOFF_EBNO = -2.0

VARIANTS = [
    "main_d256_b4_r2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
]

UPAIR_RECEIVER = "upair5g_lmmse"
BENCHMARKS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "perfect_csi_lmmse",
]

SHORT = {
    "main_d256_b4_r2": "Main",
    "shallow_d256_b2_r2": "Shallow",
    "deep_d256_b6_r2": "Deep",
    "narrow_d192_b4_r2": "Narrow",
    "wide_d320_b4_r2": "Wide",
    "wide_deep_d320_b6_r2": "Wide-deep",
    "mlpwide_d256_b4_r4": "MLP-wide",
    "baseline_ls_lmmse": "LS",
    "baseline_ls_2dlmmse_lmmse": "2D-LMMSE",
    "perfect_csi_lmmse": "Perfect CSI",
}


LEGEND = {
    "main_d256_b4_r2": "UPAIR Main (d=256, L=4, r=2)",
    "shallow_d256_b2_r2": "UPAIR Shallow (d=256, L=2, r=2)",
    "deep_d256_b6_r2": "UPAIR Deep (d=256, L=6, r=2)",
    "narrow_d192_b4_r2": "UPAIR Narrow (d=192, L=4, r=2)",
    "wide_d320_b4_r2": "UPAIR Wide (d=320, L=4, r=2)",
    "wide_deep_d320_b6_r2": "UPAIR Wide-deep (d=320, L=6, r=2)",
    "mlpwide_d256_b4_r4": "UPAIR MLP-wide (d=256, L=4, r=4)",
    "baseline_ls_lmmse": "LS estimator + LMMSE detector",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D-LMMSE estimator + LMMSE detector",
    "perfect_csi_lmmse": "Perfect CSI + LMMSE detector",
}

METHOD_ORDER = VARIANTS + BENCHMARKS
UPAIR_MARKERS = ["o", "s", "^", "v", "D", "P", "X"]
BENCH_MARKERS = {
    "baseline_ls_lmmse": "<",
    "baseline_ls_2dlmmse_lmmse": ">",
    "perfect_csi_lmmse": "*",
}
BENCH_LINESTYLES = {
    "baseline_ls_lmmse": "--",
    "baseline_ls_2dlmmse_lmmse": "-.",
    "perfect_csi_lmmse": ":",
}

COUNT_COLS = [
    "bit_errors",
    "num_bits",
    "block_errors",
    "num_blocks",
    "num_batches_run",
]
TIME_COLS = [
    "point_elapsed_s",
    "data_elapsed_s",
    "receiver_elapsed_s",
]
KEY_COLS = ["variant", "receiver", "num_users", "ebno_db"]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 13,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.65,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root (default: current directory)",
    )
    return parser.parse_args()


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")


def _in_eval_grid(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    for ebno in EBNOS:
        mask |= np.isclose(values, ebno, atol=1e-9, rtol=0.0)
    return mask


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in ("variant", "receiver"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].astype(str)

    _numeric(
        frame,
        [
            "num_users",
            "ebno_db",
            "chunk_idx",
            "nmse",
            *COUNT_COLS,
            *TIME_COLS,
        ],
    )
    return frame.dropna(subset=KEY_COLS)


def _chunk_idx_from_path(path: Path) -> int:
    match = re.search(r"chunk(\d+)", str(path))
    if match:
        return int(match.group(1))
    return abs(hash(str(path))) % (2**31)


def _finalize_eval_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for column in COUNT_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
        out[column] = out[column].round().astype("int64")
    for column in TIME_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["ber"] = np.where(
        out["num_bits"] > 0,
        out["bit_errors"] / out["num_bits"],
        np.nan,
    )
    out["bler"] = np.where(
        out["num_blocks"] > 0,
        out["block_errors"] / out["num_blocks"],
        np.nan,
    )
    out["receiver_ms_per_batch"] = np.where(
        (out["receiver_elapsed_s"] > 0) & (out["num_batches_run"] > 0),
        1000.0 * out["receiver_elapsed_s"] / out["num_batches_run"],
        np.nan,
    )
    out["receiver_ms_per_frame"] = np.where(
        (out["receiver_elapsed_s"] > 0) & (out["num_blocks"] > 0),
        1000.0 * out["receiver_elapsed_s"] / out["num_blocks"],
        np.nan,
    )
    out["reliable"] = out["block_errors"] >= TARGET_BLOCK_ERRORS
    out["done"] = (
        ((out["num_batches_run"] >= MIN_BATCHES) & out["reliable"])
        | (out["num_batches_run"] >= MAX_BATCHES)
    )
    out["source"] = source
    return out


def load_raw_chunks(root: Path, receivers: set[str]) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("chunk_result.csv")):
        try:
            frame = _normalize_frame(pd.read_csv(path))
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue
        if frame.empty:
            continue
        frame["_mtime"] = path.stat().st_mtime
        frame["_path"] = str(path)
        if frame["chunk_idx"].isna().all():
            frame["chunk_idx"] = _chunk_idx_from_path(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data[
        data["receiver"].isin(receivers)
        & data["num_users"].eq(U)
        & _in_eval_grid(data["ebno_db"])
    ].copy()
    if data.empty:
        return data

    data["chunk_idx"] = data["chunk_idx"].fillna(-1).astype(int)
    data["_rank_blocks"] = pd.to_numeric(
        data["num_blocks"], errors="coerce"
    ).fillna(-1)
    data = (
        data.sort_values(
            KEY_COLS + ["chunk_idx", "_mtime", "_rank_blocks"],
            ascending=[True, True, True, True, True, True, True],
        )
        .drop_duplicates(KEY_COLS + ["chunk_idx"], keep="last")
        .copy()
    )

    rows: list[dict[str, Any]] = []
    for key, group in data.groupby(KEY_COLS, dropna=False):
        row = dict(zip(KEY_COLS, key, strict=False))
        for column in COUNT_COLS:
            row[column] = float(
                pd.to_numeric(group[column], errors="coerce").fillna(0).sum()
            )
        for column in TIME_COLS:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.fillna(0).sum()) if values.notna().any() else np.nan

        nmse = pd.to_numeric(group["nmse"], errors="coerce")
        weights = pd.to_numeric(group["num_batches_run"], errors="coerce").fillna(0)
        valid_nmse = nmse.notna() & np.isfinite(nmse) & (weights > 0)
        if valid_nmse.any():
            row["nmse"] = float(
                np.average(nmse[valid_nmse], weights=weights[valid_nmse])
            )
        else:
            row["nmse"] = np.nan
        rows.append(row)

    return _finalize_eval_rows(pd.DataFrame(rows), "raw chunks")


def load_merged_csvs(root: Path, receivers: set[str]) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("merged_*.csv")):
        try:
            frame = _normalize_frame(pd.read_csv(path))
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue
        if frame.empty:
            continue
        frame["_mtime"] = path.stat().st_mtime
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data[
        data["receiver"].isin(receivers)
        & data["num_users"].eq(U)
        & _in_eval_grid(data["ebno_db"])
    ].copy()
    if data.empty:
        return data

    data["_rank_blocks"] = pd.to_numeric(
        data["num_blocks"], errors="coerce"
    ).fillna(-1)
    data = (
        data.sort_values(
            KEY_COLS + ["_rank_blocks", "_mtime"],
            ascending=[True, True, True, True, True, True],
        )
        .drop_duplicates(KEY_COLS, keep="last")
        .copy()
    )
    keep = KEY_COLS + COUNT_COLS + TIME_COLS + ["nmse"]
    return _finalize_eval_rows(data[keep], "merged CSV")


def combine_raw_and_merged(raw: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not raw.empty:
        raw = raw.copy()
        raw["_priority"] = 0
        frames.append(raw)
    if not merged.empty:
        merged = merged.copy()
        merged["_priority"] = 1
        frames.append(merged)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["_rank_blocks"] = pd.to_numeric(
        data["num_blocks"], errors="coerce"
    ).fillna(-1)
    data = (
        data.sort_values(
            KEY_COLS + ["_priority", "_rank_blocks"],
            ascending=[True, True, True, True, True, False],
        )
        .drop_duplicates(KEY_COLS, keep="first")
        .drop(columns=["_priority", "_rank_blocks"])
    )
    return data


def method_key(row: pd.Series) -> str:
    return str(row["variant"]) if row["receiver"] == UPAIR_RECEIVER else str(row["receiver"])


def method_style(method: str) -> tuple[str, str, str]:
    if method in VARIANTS:
        index = VARIANTS.index(method)
        return f"C{index}", "-", UPAIR_MARKERS[index]
    index = BENCHMARKS.index(method)
    return f"C{7 + index}", BENCH_LINESTYLES[method], BENCH_MARKERS[method]


def save_figure(fig: plt.Figure, out: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for extension in ("png", "pdf", "svg"):
        path = out / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=350 if extension == "png" else None,
            bbox_inches="tight",
        )
        paths.append(path)
    plt.close(fig)
    return paths


def plot_bler(
    data: pd.DataFrame,
    methods: list[str],
    out: Path,
    stem: str,
    title: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    plotted = 0

    for method in methods:
        subset = data[data["method"].eq(method)].sort_values("ebno_db")
        if subset.empty:
            continue

        point_map = {float(row.ebno_db): row for row in subset.itertuples(index=False)}
        y = np.array(
            [
                float(point_map[x].bler)
                if x in point_map
                and np.isfinite(point_map[x].bler)
                and point_map[x].bler > 0
                else np.nan
                for x in EBNOS
            ],
            dtype=float,
        )
        if not np.isfinite(y).any():
            continue

        color, linestyle, marker = method_style(method)
        ax.plot(
            EBNOS,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=LEGEND[method],
            zorder=2,
        )

        points = subset[np.isfinite(subset["bler"]) & subset["bler"].gt(0)].copy()
        x_values = points["ebno_db"].to_numpy(dtype=float)
        y_values = points["bler"].to_numpy(dtype=float)
        reliable = points["reliable"].astype(bool).to_numpy()
        complete_low_count = points["done"].astype(bool).to_numpy() & ~reliable
        partial = ~points["done"].astype(bool).to_numpy()

        if reliable.any():
            ax.scatter(
                x_values[reliable],
                y_values[reliable],
                marker=marker,
                s=54,
                color=color,
                zorder=4,
            )
        if complete_low_count.any():
            ax.scatter(
                x_values[complete_low_count],
                y_values[complete_low_count],
                marker=marker,
                s=58,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                zorder=4,
            )
        if partial.any():
            ax.scatter(
                x_values[partial],
                y_values[partial],
                marker="x",
                s=58,
                color=color,
                linewidths=1.5,
                zorder=5,
            )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return []

    ax.set_yscale("log")
    ax.set_xlim(min(EBNOS) - 0.2, max(EBNOS) + 0.2)
    ax.set_xticks(EBNOS)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("BLER")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    return save_figure(fig, out, stem)


def plot_bler_gain(data: pd.DataFrame, out: Path) -> list[Path]:
    baseline = data[
        data["method"].eq("baseline_ls_2dlmmse_lmmse")
        & np.isfinite(data["bler"])
        & data["bler"].gt(0)
    ].set_index("ebno_db")
    if baseline.empty:
        return []

    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    plotted = 0
    for method in VARIANTS:
        current = data[
            data["method"].eq(method)
            & np.isfinite(data["bler"])
            & data["bler"].gt(0)
        ].set_index("ebno_db")
        common = sorted(set(current.index.astype(float)) & set(baseline.index.astype(float)))
        if not common:
            continue
        gain = [float(baseline.loc[x, "bler"] / current.loc[x, "bler"]) for x in common]
        color, _, marker = method_style(method)
        ax.plot(
            common,
            gain,
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=5.5,
            label=LEGEND[method],
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return []

    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xticks(EBNOS)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("BLER gain")
    ax.set_title("BLER gain over 2D-LMMSE (3 users)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    return save_figure(fig, out, "Fig05_bler_gain_over_2dlmmse_u3")


def timing_summary(data: pd.DataFrame) -> pd.DataFrame:
    timed = data[
        data["done"].astype(bool)
        & np.isfinite(data["receiver_elapsed_s"])
        & data["receiver_elapsed_s"].gt(0)
        & data["num_blocks"].gt(0)
        & data["num_batches_run"].gt(0)
    ].copy()
    if timed.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for method, group in timed.groupby("method", sort=False):
        receiver_s = float(group["receiver_elapsed_s"].sum())
        blocks = int(group["num_blocks"].sum())
        batches = int(group["num_batches_run"].sum())
        rows.append(
            {
                "method": method,
                "label": SHORT.get(method, method),
                "num_snr_points": int(len(group)),
                "receiver_elapsed_s": receiver_s,
                "num_blocks": blocks,
                "num_batches_run": batches,
                "pooled_ms_per_frame": 1000.0 * receiver_s / max(blocks, 1),
                "pooled_ms_per_batch": 1000.0 * receiver_s / max(batches, 1),
                "median_point_ms_per_frame": float(group["receiver_ms_per_frame"].median()),
                "min_point_ms_per_frame": float(group["receiver_ms_per_frame"].min()),
                "max_point_ms_per_frame": float(group["receiver_ms_per_frame"].max()),
            }
        )
    result = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    result["_order"] = result["method"].map(order).fillna(999)
    return result.sort_values("_order").drop(columns="_order")


def plot_timing_bar(summary: pd.DataFrame, out: Path) -> list[Path]:
    if summary.empty:
        return []
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    x = np.arange(len(summary))
    values = summary["pooled_ms_per_frame"].to_numpy(dtype=float)
    colors = [method_style(method)[0] for method in summary["method"]]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    positive = values[values > 0]
    if len(positive) and positive.max() / positive.min() > 15:
        ax.set_yscale("log")
    ax.set_xticks(x, summary["label"], rotation=30, ha="right")
    ax.set_ylabel("Receiver time (ms / block)")
    ax.set_title("Receiver latency (3 users)")
    fig.tight_layout()
    return save_figure(fig, out, "Fig06_receiver_latency_u3")


def plot_timing_vs_ebno(data: pd.DataFrame, out: Path) -> list[Path]:
    timed = data[
        data["done"].astype(bool)
        & np.isfinite(data["receiver_ms_per_frame"])
        & data["receiver_ms_per_frame"].gt(0)
    ].copy()
    if timed.empty:
        return []

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    plotted = 0
    for method in METHOD_ORDER:
        subset = timed[timed["method"].eq(method)].sort_values("ebno_db")
        if subset.empty:
            continue
        color, linestyle, marker = method_style(method)
        ax.plot(
            subset["ebno_db"],
            subset["receiver_ms_per_frame"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=LEGEND[method],
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []

    values = timed["receiver_ms_per_frame"].to_numpy(dtype=float)
    if values.max() / values.min() > 15:
        ax.set_yscale("log")
    ax.set_xticks(EBNOS)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("Receiver time (ms / block)")
    ax.set_title("Receiver latency by SNR (3 users)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    return save_figure(fig, out, "Fig07_receiver_latency_vs_ebno_u3")


def plot_bler_latency_tradeoff(data: pd.DataFrame, out: Path) -> list[Path]:
    points = data[
        np.isclose(data["ebno_db"], TRADEOFF_EBNO)
        & data["done"].astype(bool)
        & np.isfinite(data["bler"])
        & data["bler"].gt(0)
        & np.isfinite(data["receiver_ms_per_frame"])
        & data["receiver_ms_per_frame"].gt(0)
    ].copy()
    if points.empty:
        return []

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    for row in points.itertuples(index=False):
        color, _, marker = method_style(row.method)
        ax.scatter(
            row.receiver_ms_per_frame,
            row.bler,
            color=color,
            marker=marker,
            s=68,
            zorder=3,
        )
        ax.annotate(
            SHORT[row.method],
            (row.receiver_ms_per_frame, row.bler),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Receiver time (ms / block)")
    ax.set_ylabel("BLER")
    ax.set_title(r"BLER–latency tradeoff at $E_b/N_0=-2$ dB")
    fig.tight_layout()
    return save_figure(fig, out, "Fig08_bler_latency_tradeoff_m2dB_u3")


def plot_eval_nmse(data: pd.DataFrame, out: Path) -> tuple[list[Path], bool]:
    # Do not create a misleading plot from Perfect CSI alone. Require at least
    # two non-perfect methods with finite, positive evaluation NMSE.
    nmse = data[
        data["method"].ne("perfect_csi_lmmse")
        & np.isfinite(data["nmse"])
        & data["nmse"].gt(0)
    ].copy()
    if nmse["method"].nunique() < 2:
        return [], False

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    for method in METHOD_ORDER:
        subset = nmse[nmse["method"].eq(method)].sort_values("ebno_db")
        if subset.empty:
            continue
        color, linestyle, marker = method_style(method)
        ax.plot(
            subset["ebno_db"],
            subset["nmse"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            label=LEGEND[method],
        )
    ax.set_yscale("log")
    ax.set_xticks(EBNOS)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("NMSE")
    ax.set_title("Evaluation NMSE (3 users)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    return save_figure(fig, out, "Fig09_evaluation_nmse_u3"), True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return {}


def load_training_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = root / "TWC_plots_comprehensive" / "runs_rx16" / "seed7" / "1dmrs"
    summaries: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    for variant in VARIANTS:
        metrics = run_root / variant / "metrics"
        state = _read_json(metrics / "train_state.json")
        model = _read_json(metrics / "model_summary.json")
        history_payload = _read_json(metrics / "history.json")
        history = history_payload.get("history", [])
        if not isinstance(history, list):
            history = []

        # Keep the last copy of a step if a resume produced duplicates.
        by_step: dict[int, dict[str, Any]] = {}
        for item in history:
            if not isinstance(item, dict):
                continue
            try:
                step = int(item.get("step"))
            except Exception:
                continue
            by_step[step] = item
        rows = [by_step[step] for step in sorted(by_step)]

        step_times: list[float] = []
        total_logged_s = 0.0
        validation_rows: list[dict[str, Any]] = []
        for item in rows:
            try:
                step = int(item["step"])
            except Exception:
                continue
            elapsed = pd.to_numeric(item.get("step_elapsed_s"), errors="coerce")
            if np.isfinite(elapsed) and elapsed >= 0:
                total_logged_s += float(elapsed)

            val_prop = pd.to_numeric(item.get("val_nmse_prop"), errors="coerce")
            val_ls = pd.to_numeric(item.get("val_nmse_ls"), errors="coerce")
            has_validation = np.isfinite(val_prop) and np.isfinite(val_ls) and val_ls > 0
            if has_validation:
                record = {
                    "variant": variant,
                    "step": step,
                    "val_nmse_prop": float(val_prop),
                    "val_nmse_ls": float(val_ls),
                    "val_nmse_ratio": float(val_prop / val_ls),
                }
                validation_rows.append(record)
                validations.append(record)
            elif step > 100 and np.isfinite(elapsed) and elapsed > 0:
                step_times.append(float(elapsed))

        final_validation = max(validation_rows, key=lambda row: row["step"], default=None)
        best_validation = min(
            validation_rows,
            key=lambda row: row["val_nmse_prop"],
            default=None,
        )
        summaries.append(
            {
                "variant": variant,
                "label": SHORT[variant],
                "training_complete": bool(state.get("training_complete", False)),
                "latest_step": int(state.get("latest_step", -1)),
                "total_steps": int(state.get("total_steps", 40000)),
                "num_trainable_params": pd.to_numeric(
                    model.get("num_trainable_params"), errors="coerce"
                ),
                "logged_training_hours": total_logged_s / 3600.0 if total_logged_s > 0 else np.nan,
                "median_regular_step_ms": 1000.0 * float(np.median(step_times)) if step_times else np.nan,
                "p90_regular_step_ms": 1000.0 * float(np.percentile(step_times, 90)) if step_times else np.nan,
                "final_val_nmse": final_validation["val_nmse_prop"] if final_validation else np.nan,
                "final_val_nmse_ls": final_validation["val_nmse_ls"] if final_validation else np.nan,
                "final_val_nmse_ratio": final_validation["val_nmse_ratio"] if final_validation else np.nan,
                "best_val_nmse": best_validation["val_nmse_prop"] if best_validation else np.nan,
                "best_val_nmse_ratio": best_validation["val_nmse_ratio"] if best_validation else np.nan,
            }
        )

    return pd.DataFrame(summaries), pd.DataFrame(validations)


def plot_validation_nmse_ratio(validation: pd.DataFrame, out: Path) -> list[Path]:
    if validation.empty:
        return []
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    plotted = 0
    for variant in VARIANTS:
        subset = validation[validation["variant"].eq(variant)].sort_values("step")
        if subset.empty:
            continue
        color, _, marker = method_style(variant)
        ax.plot(
            subset["step"] / 1000.0,
            subset["val_nmse_ratio"],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            label=LEGEND[variant],
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []
    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("UPAIR NMSE / LS NMSE")
    ax.set_title("Validation NMSE ratio")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    return save_figure(fig, out, "Fig10_validation_nmse_ratio")


def plot_training_runtime(summary: pd.DataFrame, out: Path) -> list[Path]:
    usable = summary[np.isfinite(summary["logged_training_hours"])].copy()
    if usable.empty:
        return []
    order = {name: index for index, name in enumerate(VARIANTS)}
    usable["_order"] = usable["variant"].map(order)
    usable = usable.sort_values("_order")
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    colors = [method_style(variant)[0] for variant in usable["variant"]]
    ax.bar(
        np.arange(len(usable)),
        usable["logged_training_hours"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(np.arange(len(usable)), usable["label"], rotation=25, ha="right")
    ax.set_ylabel("Logged training time (h)")
    ax.set_title("Training runtime")
    fig.tight_layout()
    return save_figure(fig, out, "Fig11_training_runtime")


def plot_training_step_time(summary: pd.DataFrame, out: Path) -> list[Path]:
    usable = summary[np.isfinite(summary["median_regular_step_ms"])].copy()
    if usable.empty:
        return []
    order = {name: index for index, name in enumerate(VARIANTS)}
    usable["_order"] = usable["variant"].map(order)
    usable = usable.sort_values("_order")
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    colors = [method_style(variant)[0] for variant in usable["variant"]]
    ax.bar(
        np.arange(len(usable)),
        usable["median_regular_step_ms"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(np.arange(len(usable)), usable["label"], rotation=25, ha="right")
    ax.set_ylabel("Median step time (ms)")
    ax.set_title("Training step time")
    fig.tight_layout()
    return save_figure(fig, out, "Fig12_training_step_time")


def plot_size_vs_nmse(summary: pd.DataFrame, out: Path) -> list[Path]:
    usable = summary[
        np.isfinite(summary["num_trainable_params"])
        & np.isfinite(summary["final_val_nmse_ratio"])
        & summary["num_trainable_params"].gt(0)
    ].copy()
    if usable.empty:
        return []
    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    for row in usable.itertuples(index=False):
        color, _, marker = method_style(row.variant)
        x = row.num_trainable_params / 1e6
        y = row.final_val_nmse_ratio
        ax.scatter(x, y, color=color, marker=marker, s=72)
        ax.annotate(
            row.label,
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel("Trainable parameters (million)")
    ax.set_ylabel("Final UPAIR NMSE / LS NMSE")
    ax.set_title("Model size vs validation NMSE")
    fig.tight_layout()
    return save_figure(fig, out, "Fig13_model_size_vs_validation_nmse")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not (root / "src" / "upair5g").exists():
        raise SystemExit(f"Not a UPAIR repository root: {root}")

    out = root / "BLER_3u_eval-results_comprehensive"
    out.mkdir(parents=True, exist_ok=True)

    upair_root = root / "_isolated_eval_chunks"
    baseline_root = root / "_final_u3_baseline_chunks"

    upair = combine_raw_and_merged(
        load_raw_chunks(upair_root, {UPAIR_RECEIVER}),
        load_merged_csvs(upair_root, {UPAIR_RECEIVER}),
    )
    if not upair.empty:
        upair = upair[
            upair["variant"].isin(VARIANTS)
            & upair["receiver"].eq(UPAIR_RECEIVER)
        ].copy()

    baselines = combine_raw_and_merged(
        load_raw_chunks(baseline_root, set(BENCHMARKS)),
        load_merged_csvs(baseline_root, set(BENCHMARKS)),
    )
    if not baselines.empty:
        baselines = baselines[
            baselines["receiver"].isin(BENCHMARKS)
        ].copy()
        # Benchmarks are architecture-independent. Keep the main variant tag
        # used by the final baseline worker only as bookkeeping.
        baselines = baselines[baselines["variant"].eq("main_d256_b4_r2")].copy()

    data = pd.concat([upair, baselines], ignore_index=True, sort=False)
    if data.empty:
        raise SystemExit("No 3-user evaluation data found.")
    data["method"] = data.apply(method_key, axis=1)
    data["label"] = data["method"].map(SHORT).fillna(data["method"])
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    data["_order"] = data["method"].map(order).fillna(999)
    data = data.sort_values(["_order", "ebno_db"]).drop(columns="_order")

    data.to_csv(out / "evaluation_results_all.csv", index=False)
    data[data["bler"].gt(0)].to_csv(out / "bler_positive_points.csv", index=False)
    data[data["bler"].eq(0)].to_csv(out / "bler_zero_points_omitted.csv", index=False)

    generated: list[Path] = []
    generated += plot_bler(
        data,
        METHOD_ORDER,
        out,
        "Fig01_bler_all_methods_u3",
        "BLER comparison (3 users)",
    )
    generated += plot_bler(
        data,
        VARIANTS,
        out,
        "Fig02_bler_upair_variants_u3",
        "UPAIR variants (3 users)",
    )
    generated += plot_bler(
        data,
        ["main_d256_b4_r2", *BENCHMARKS],
        out,
        "Fig03_bler_main_vs_benchmarks_u3",
        "Main UPAIR vs benchmarks (3 users)",
    )
    generated += plot_bler(
        data,
        ["wide_deep_d320_b6_r2", *BENCHMARKS],
        out,
        "Fig04_bler_wide_deep_vs_benchmarks_u3",
        "Wide-deep UPAIR ($d$=320, $L$=6, $r$=2) vs benchmarks",
    )
    generated += plot_bler_gain(data, out)

    timing = timing_summary(data)
    timing.to_csv(out / "receiver_timing_summary.csv", index=False)
    data[
        np.isfinite(data["receiver_ms_per_frame"])
        & data["receiver_ms_per_frame"].gt(0)
    ].to_csv(out / "receiver_timing_by_snr.csv", index=False)
    generated += plot_timing_bar(timing, out)
    generated += plot_timing_vs_ebno(data, out)
    generated += plot_bler_latency_tradeoff(data, out)

    nmse_paths, eval_nmse_available = plot_eval_nmse(data, out)
    generated += nmse_paths

    training_summary, validation = load_training_data(root)
    training_summary.to_csv(out / "training_summary.csv", index=False)
    validation.to_csv(out / "validation_nmse_history.csv", index=False)
    generated += plot_validation_nmse_ratio(validation, out)
    generated += plot_training_runtime(training_summary, out)
    generated += plot_training_step_time(training_summary, out)
    generated += plot_size_vs_nmse(training_summary, out)

    expected_rows: list[dict[str, Any]] = []
    lookup = {
        (row.method, float(row.ebno_db)): row
        for row in data.itertuples(index=False)
    }
    for method in METHOD_ORDER:
        for ebno in EBNOS:
            row = lookup.get((method, ebno))
            expected_rows.append(
                {
                    "method": method,
                    "label": SHORT[method],
                    "ebno_db": ebno,
                    "available": row is not None,
                    "done": bool(row.done) if row is not None else False,
                    "reliable": bool(row.reliable) if row is not None else False,
                    "bler": float(row.bler) if row is not None else np.nan,
                    "block_errors": int(row.block_errors) if row is not None else 0,
                    "num_batches_run": int(row.num_batches_run) if row is not None else 0,
                }
            )
    audit = pd.DataFrame(expected_rows)
    audit.to_csv(out / "evaluation_coverage.csv", index=False)

    readme = [
        "Comprehensive 3-user UPAIR results",
        "",
        "BLER:",
        "- BLER=0, NaN, and unavailable points are omitted from BLER figures.",
        "- Filled markers have at least 100 block errors.",
        "- Open markers completed at the 2000-batch cap with fewer than 100 errors.",
        "- An x marker denotes a partial point.",
        "",
        "Timing:",
        "- Receiver timing uses receiver_elapsed_s only: estimator plus detector/decoder path.",
        "- Data generation, initialization, warm-up, and shared covariance construction are excluded.",
        "- Values describe the recorded H100 software runs; they are not hardware-independent complexity measures.",
        "",
        "NMSE:",
        (
            "- Evaluation NMSE was available and Fig09 was generated."
            if eval_nmse_available
            else "- Evaluation NMSE was not collected for the BLER-only isolated runs; no evaluation-NMSE figure was generated."
        ),
        "- Training validation NMSE is available and is shown as UPAIR NMSE / LS NMSE.",
        "- Validation sampled users 1-4 and the configured validation SNR grid; it is not a 3-user test-NMSE curve.",
        "",
        "Training timing:",
        "- Logged training time is the sum of step_elapsed_s and excludes queueing/downtime between resumed jobs.",
        "- Median step time excludes validation rows and the first 100 warm-up steps.",
        "",
        "Generated figures:",
        *[f"- {path.name}" for path in generated if path.suffix == ".png"],
    ]
    (out / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[OK] Output directory: {out}")
    print(f"[OK] Evaluation rows: {len(data)}")
    print(f"[OK] Completed expected points: {int(audit['done'].sum())}/{len(audit)}")
    print(
        "[INFO] Evaluation NMSE figure:",
        "generated" if eval_nmse_available else "not generated (NMSE unavailable)",
    )
    print(f"[OK] PNG figures: {sum(path.suffix == '.png' for path in generated)}")


if __name__ == "__main__":
    main()
