#!/usr/bin/env python3
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "_isolated_eval_chunks"

LABEL = {
    "baseline_ls_lmmse": "LS + LMMSE detector",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D-LMMSE + LMMSE detector",
    "perfect_csi_lmmse": "Perfect CSI + LMMSE detector",
}

BENCH_ORDER = {
    "baseline_ls_lmmse": 0,
    "baseline_ls_2dlmmse_lmmse": 1,
    "perfect_csi_lmmse": 2,
}

BENCH_STYLE = {
    "baseline_ls_lmmse": ("--", "s", 2.2),
    "baseline_ls_2dlmmse_lmmse": ("-.", "D", 2.2),
    "perfect_csi_lmmse": (":", "*", 2.4),
}

MARKERS = ["o", "^", "v", "P", "X", "<", ">", "h", "d"]

VARIANT_ORDER = {
    "main_d256_b4_r2": 0,
    "shallow_d256_b2_r2": 1,
    "deep_d256_b6_r2": 2,
    "narrow_d192_b4_r2": 3,
    "wide_d320_b4_r2": 4,
    "wide_deep_d320_b6_r2": 5,
    "mlpwide_d256_b4_r4": 6,
}


def bool_col(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    return series.astype(str).str.lower().map({
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }).astype("boolean")


def is_upair(receiver):
    return str(receiver).lower().startswith("upair")


def variant_label(name):
    families = [
        ("wide_deep", "Wide-deep"),
        ("mlpwide", "MLP-wide"),
        ("shallow", "Shallow"),
        ("narrow", "Narrow"),
        ("deep", "Deep"),
        ("wide", "Wide"),
        ("main", "Main"),
    ]

    family = next((label for prefix, label in families
                   if name.startswith(prefix)), name)

    match = re.search(r"d(\d+)_b(\d+)_r(\d+)", name)
    if match:
        return (
            f"Extended UPAIR {family} "
            f"(d={match[1]}, L={match[2]}, r={match[3]})"
        )

    return f"Extended UPAIR {name}"


def load_data():
    paths = sorted(SRC.glob("merged_*.csv"))

    if not paths:
        raise SystemExit(f"No merged_*.csv files found in {SRC}")

    frames = []

    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] skipped {path.name}: {exc}")
            continue

        frame["source"] = str(path.relative_to(ROOT))
        frame["mtime"] = path.stat().st_mtime
        frames.append(frame)

    if not frames:
        raise SystemExit("No readable merged CSV files.")

    data = pd.concat(frames, ignore_index=True, sort=False)

    required = {"variant", "receiver", "num_users", "ebno_db", "bler"}
    if not required.issubset(data.columns):
        missing = sorted(required - set(data.columns))
        raise SystemExit(f"Missing required columns: {missing}")

    numeric_columns = [
        "num_users",
        "ebno_db",
        "bler",
        "ber",
        "block_errors",
        "num_blocks",
        "bit_errors",
        "num_bits",
    ]

    for column in numeric_columns:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=["variant", "receiver", "num_users", "ebno_db", "bler"]
    )

    data["num_users"] = data["num_users"].astype(int)
    data["variant"] = data["variant"].astype(str)
    data["receiver"] = data["receiver"].astype(str)

    if "reliable_bler" in data:
        data["reliable_bler"] = bool_col(
            data["reliable_bler"]
        ).fillna(data["block_errors"] >= 100)
    else:
        data["reliable_bler"] = data["block_errors"] >= 100

    if "reliable_ber" in data:
        data["reliable_ber"] = bool_col(
            data["reliable_ber"]
        ).fillna(data["bit_errors"] >= 1000)
    else:
        data["reliable_ber"] = data["bit_errors"] >= 1000

    # If duplicate merged outputs exist, retain the most complete one.
    data["rank_blocks"] = data["num_blocks"].fillna(-1)

    data = (
        data.sort_values(
            [
                "variant",
                "receiver",
                "num_users",
                "ebno_db",
                "rank_blocks",
                "mtime",
            ],
            ascending=[True, True, True, True, False, False],
        )
        .drop_duplicates(
            ["variant", "receiver", "num_users", "ebno_db"]
        )
        .drop(columns="rank_blocks")
    )

    upair = data[data["receiver"].map(is_upair)].copy()
    benchmarks = data[~data["receiver"].map(is_upair)].copy()

    if not benchmarks.empty:
        # Benchmarks do not depend on UPAIR architecture.
        # Prefer the benchmark evaluated with the main variant.
        benchmarks["prefer_main"] = benchmarks["variant"].eq(
            "main_d256_b4_r2"
        ).astype(int)

        benchmarks["rank_blocks"] = benchmarks["num_blocks"].fillna(-1)

        benchmarks = (
            benchmarks.sort_values(
                [
                    "receiver",
                    "num_users",
                    "ebno_db",
                    "prefer_main",
                    "rank_blocks",
                    "mtime",
                ],
                ascending=[True, True, True, False, False, False],
            )
            .drop_duplicates(["receiver", "num_users", "ebno_db"])
            .drop(columns=["prefer_main", "rank_blocks"])
        )

    selected = pd.concat(
        [upair, benchmarks],
        ignore_index=True,
        sort=False,
    )

    return paths, data, selected


def specs(frame, include_upair=True, include_benchmarks=True,
          only_variant=None):

    output = []

    if include_upair:
        variants = sorted(
            frame.loc[
                frame["receiver"].map(is_upair), "variant"
            ].unique(),
            key=lambda value: (
                VARIANT_ORDER.get(value, 99),
                value,
            ),
        )

        if only_variant:
            variants = [
                variant for variant in variants
                if variant == only_variant
            ]

        for index, variant in enumerate(variants):
            output.append((
                frame["receiver"].map(is_upair)
                & frame["variant"].eq(variant),
                variant_label(variant),
                "-",
                MARKERS[index % len(MARKERS)],
                1.8,
            ))

    if include_benchmarks:
        receivers = sorted(
            frame.loc[
                ~frame["receiver"].map(is_upair), "receiver"
            ].unique(),
            key=lambda value: (
                BENCH_ORDER.get(value, 99),
                value,
            ),
        )

        for receiver in receivers:
            linestyle, marker, linewidth = BENCH_STYLE.get(
                receiver,
                ("--", "s", 2.1),
            )

            output.append((
                frame["receiver"].eq(receiver),
                LABEL.get(receiver, receiver),
                linestyle,
                marker,
                linewidth,
            ))

    return output


def make_plot(
    frame,
    users,
    metric,
    stem,
    title,
    include_upair=True,
    include_benchmarks=True,
    only_variant=None,
):
    positive = frame[
        np.isfinite(frame[metric]) & (frame[metric] > 0)
    ]

    if positive.empty:
        return []

    # Include zero-valued evaluated SNRs internally so that
    # curves break rather than connect across zero/missing points.
    grid = np.array(sorted(frame["ebno_db"].unique()), dtype=float)

    # Only SNRs having at least one positive result appear as ticks.
    ticks = np.array(
        sorted(positive["ebno_db"].unique()),
        dtype=float,
    )

    reliability_column = (
        "reliable_bler" if metric == "bler"
        else "reliable_ber"
    )

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    number_plotted = 0

    for mask, label, linestyle, marker, linewidth in specs(
        frame,
        include_upair=include_upair,
        include_benchmarks=include_benchmarks,
        only_variant=only_variant,
    ):
        series = frame.loc[mask].sort_values("ebno_db")
        series = series[
            np.isfinite(series[metric]) & (series[metric] > 0)
        ]

        if series.empty:
            continue

        values = dict(zip(
            series["ebno_db"].astype(float),
            series[metric].astype(float),
        ))

        reliability = dict(zip(
            series["ebno_db"].astype(float),
            series[reliability_column].astype(bool),
        ))

        y = np.array(
            [values.get(float(x), np.nan) for x in grid],
            dtype=float,
        )

        line, = ax.plot(
            grid,
            y,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=2,
        )

        valid = np.isfinite(y)

        reliable = np.array([
            valid[index]
            and reliability.get(float(x), False)
            for index, x in enumerate(grid)
        ])

        unreliable = valid & ~reliable

        if reliable.any():
            ax.scatter(
                grid[reliable],
                y[reliable],
                marker=marker,
                s=52,
                color=line.get_color(),
                zorder=3,
            )

        if unreliable.any():
            ax.scatter(
                grid[unreliable],
                y[unreliable],
                marker=marker,
                s=58,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.5,
                zorder=3,
            )

        number_plotted += 1

    if number_plotted == 0:
        plt.close(fig)
        return []

    ax.set_yscale("log")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel(metric.upper())

    ax.set_title(
        f"{title}\n"
        f"{users} active users — only evaluated, positive points are shown"
    )

    ax.set_xticks(ticks)

    if len(ticks) == 1:
        ax.set_xlim(ticks[0] - 0.5, ticks[0] + 0.5)
    else:
        ax.set_xlim(ticks.min() - 0.25, ticks.max() + 0.25)

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.7,
        alpha=0.7,
    )

    ax.text(
        0.01,
        0.01,
        "Filled marker: reliability threshold met; "
        "open marker: below threshold.\n"
        "Zero/NaN/unavailable points are omitted; "
        "no extrapolation is used.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "0.75",
        },
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.5,
    )

    fig.tight_layout()

    generated = []

    for extension in ("png", "pdf"):
        path = OUT / f"{stem}_u{users}.{extension}"

        fig.savefig(
            path,
            dpi=260 if extension == "png" else None,
            bbox_inches="tight",
        )

        generated.append(path)

    plt.close(fig)
    return generated


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    paths, all_rows, rows = load_data()

    rows = rows.sort_values(
        ["num_users", "receiver", "variant", "ebno_db"]
    )

    all_rows.to_csv(
        OUT / "all_deduplicated_merged_rows.csv",
        index=False,
    )

    rows.to_csv(
        OUT / "selected_plot_rows_including_zero.csv",
        index=False,
    )

    rows[
        (rows["bler"] > 0) | (rows["ber"] > 0)
    ].to_csv(
        OUT / "selected_nonzero_plot_rows.csv",
        index=False,
    )

    generated = []

    for users in sorted(rows["num_users"].unique()):
        frame = rows[rows["num_users"].eq(users)].copy()

        # Required plot: all UPAIR variants and all available benchmarks.
        generated += make_plot(
            frame,
            users,
            "bler",
            "bler_all_upair_variants_and_benchmarks",
            "Extended UPAIR Variants and Available Benchmarks — BLER",
        )

        generated += make_plot(
            frame,
            users,
            "bler",
            "bler_upair_variants_only",
            "Extended UPAIR Architecture Comparison — BLER",
            include_benchmarks=False,
        )

        variants = sorted(
            frame.loc[
                frame["receiver"].map(is_upair), "variant"
            ].unique(),
            key=lambda value: (
                VARIANT_ORDER.get(value, 99),
                value,
            ),
        )

        representative = (
            "main_d256_b4_r2"
            if "main_d256_b4_r2" in variants
            else variants[0] if variants else None
        )

        if representative:
            generated += make_plot(
                frame,
                users,
                "bler",
                "bler_main_upair_vs_benchmarks",
                f"{variant_label(representative)} "
                "vs Available Benchmarks — BLER",
                only_variant=representative,
            )

        if (frame["ber"] > 0).any():
            generated += make_plot(
                frame,
                users,
                "ber",
                "ber_all_upair_variants_and_benchmarks",
                "Extended UPAIR Variants and "
                "Available Benchmarks — BER",
            )

    readme = [
        "Source: _isolated_eval_chunks/merged_*.csv only.",
        "Raw pipeline chunk lines and _probe3_speed_* outputs are ignored.",
        "Zero, NaN, and unavailable points are omitted.",
        "Filled BLER markers: >=100 block errors.",
        "Open BLER markers: fewer than 100 block errors.",
        "Filled BER markers: >=1000 bit errors.",
        "Open BER markers: fewer than 1000 bit errors.",
        f"Active-user counts detected: "
        f"{sorted(rows['num_users'].unique().tolist())}",
        "",
        "Generated files:",
        *[f"- {path.name}" for path in generated],
    ]

    (OUT / "README.txt").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Read {len(paths)} merged CSVs")
    print(
        "[OK] Active-user counts:",
        sorted(rows["num_users"].unique().tolist()),
    )
    print(f"[OK] Output: {OUT}")

    for path in generated:
        print("    ", path.name)


if __name__ == "__main__":
    main()
