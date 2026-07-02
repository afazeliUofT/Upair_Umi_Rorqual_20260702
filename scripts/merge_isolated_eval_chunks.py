from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUM_COLS = [
    "bit_errors",
    "num_bits",
    "block_errors",
    "num_blocks",
    "num_batches_run",
    "point_elapsed_s",
    "data_elapsed_s",
    "receiver_elapsed_s",
]


def _load_rows(input_root: Path) -> pd.DataFrame:
    paths = sorted(input_root.rglob("chunk_result.csv"))
    if not paths:
        raise FileNotFoundError(f"No chunk_result.csv files found under {input_root}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["chunk_result_path"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _merge_group(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0].to_dict()
    out: dict[str, Any] = {
        "receiver": str(first["receiver"]),
        "variant": str(first.get("variant", "")),
        "dmrs_case": str(first.get("dmrs_case", "")),
        "seed": int(first.get("seed", 7)),
        "training_seed": int(first.get("training_seed", first.get("seed", 7))),
        "num_users": int(first["num_users"]),
        "ebno_db": float(first["ebno_db"]),
        "num_chunks": int(len(group)),
        "chunk_indices": ",".join(str(int(x)) for x in sorted(group["chunk_idx"].astype(int).tolist())),
    }
    for col in SUM_COLS:
        if col in group.columns:
            out[col] = float(group[col].sum())
    # Count columns should be ints
    for col in ["bit_errors", "num_bits", "block_errors", "num_blocks", "num_batches_run"]:
        if col in out:
            out[col] = int(round(float(out[col])))

    bit_errors = int(out.get("bit_errors", 0))
    num_bits = int(out.get("num_bits", 0))
    block_errors = int(out.get("block_errors", 0))
    num_blocks = int(out.get("num_blocks", 0))
    rx_time = float(out.get("receiver_elapsed_s", 0.0))
    point_time = float(out.get("point_elapsed_s", 0.0))
    batches = int(out.get("num_batches_run", 0))

    out["ber"] = float(bit_errors / num_bits) if num_bits else np.nan
    out["bler"] = float(block_errors / num_blocks) if num_blocks else np.nan
    out["nmse"] = np.nan
    out["mc_stop_reason"] = "isolated_chunk_merge"
    out["target_bler_floor"] = np.nan
    out["bler_zero_error_upper_bound"] = float(3.0 / num_blocks) if num_blocks and block_errors == 0 else np.nan
    out["ber_zero_error_upper_bound"] = float(3.0 / num_bits) if num_bits and bit_errors == 0 else np.nan
    out["receiver_ms_per_batch"] = float(1000.0 * rx_time / max(batches, 1))
    out["receiver_ms_per_frame"] = float(1000.0 * rx_time / max(num_blocks, 1))
    out["reliable_ber"] = bool(bit_errors >= 1000)
    out["reliable_bler"] = bool(block_errors >= 100)
    # Peak memory is max over chunks; current memory is not meaningful after merge.
    for col in ["gpu_mem_current_gib", "gpu_mem_peak_gib", "gpu_mem", "peak"]:
        if col in group.columns:
            try:
                out[col] = float(group[col].max())
            except Exception:
                pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge process-isolated evaluation chunks.")
    parser.add_argument("--input-root", default="_isolated_eval_chunks")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--receiver", default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--ebno-db", type=float, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    df = _load_rows(input_root)

    if args.variant:
        df = df[df["variant"].astype(str) == args.variant]
    if args.receiver:
        df = df[df["receiver"].astype(str) == args.receiver]
    if args.num_users is not None:
        df = df[df["num_users"].astype(int) == int(args.num_users)]
    if args.ebno_db is not None:
        df = df[np.isclose(df["ebno_db"].astype(float), float(args.ebno_db))]

    if df.empty:
        raise RuntimeError("No chunk rows remain after filters.")

    group_cols = ["variant", "dmrs_case", "seed", "receiver", "num_users", "ebno_db"]
    rows = [_merge_group(g) for _, g in df.groupby(group_cols, dropna=False)]
    out_df = pd.DataFrame(rows).sort_values(["variant", "num_users", "ebno_db", "receiver"])

    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = input_root / "merged_curves.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "input_root": str(input_root),
        "output_csv": str(out_csv),
        "num_input_chunks": int(len(df)),
        "num_merged_rows": int(len(out_df)),
        "rows": rows,
    }
    summary_json = Path(args.summary_json) if args.summary_json else out_csv.with_suffix(".summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[ISO-MERGE] chunks:", len(df))
    print("[ISO-MERGE] rows:", len(out_df))
    print("[ISO-MERGE] wrote:", out_csv)
    print("[ISO-MERGE] wrote:", summary_json)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
