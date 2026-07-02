from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_float_equal(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        return False


def _rows(root: Path, variant: str, receiver: str, num_users: int, ebno_db: float) -> pd.DataFrame:
    frames = []
    for p in sorted(root.rglob("chunk_result.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0]
        if str(row.get("variant", "")) != variant:
            continue
        if str(row.get("receiver", "")) != receiver:
            continue
        try:
            if int(row.get("num_users")) != int(num_users):
                continue
        except Exception:
            continue
        if not _safe_float_equal(row.get("ebno_db"), ebno_db):
            continue
        df["chunk_result_path"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_int_set(df: pd.DataFrame) -> set[int]:
    out: set[int] = set()
    if "chunk_idx" not in df.columns:
        return out
    for x in df["chunk_idx"].tolist():
        try:
            out.add(int(x))
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="_isolated_eval_chunks")
    ap.add_argument("--config", default="configs/twc_comprehensive_mu32_base.yaml")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--num-users", type=int, required=True)
    ap.add_argument("--ebno-db", type=float, required=True)
    ap.add_argument("--chunk-batches", type=int, default=20)
    ap.add_argument("--target-block-errors", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--min-batches", type=int, default=None)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    ev = cfg.get("evaluation", {})
    target = int(args.target_block_errors if args.target_block_errors is not None else ev.get("target_block_errors_per_receiver", 100))
    max_batches = int(args.max_batches if args.max_batches is not None else ev.get("max_num_batches_per_point", 2000))
    min_batches = int(args.min_batches if args.min_batches is not None else ev.get("min_num_batches_per_point", 16))
    chunk_batches = max(1, int(args.chunk_batches))
    max_chunks = (max_batches + chunk_batches - 1) // chunk_batches

    root = Path(args.input_root)
    df = _rows(root, args.variant, args.receiver, args.num_users, args.ebno_db)
    if df.empty:
        chunks_done: set[int] = set()
        bit_errors = num_bits = block_errors = num_blocks = num_batches = 0
    else:
        chunks_done = _parse_int_set(df)
        bit_errors = int(df.get("bit_errors", pd.Series(dtype=float)).fillna(0).sum())
        num_bits = int(df.get("num_bits", pd.Series(dtype=float)).fillna(0).sum())
        block_errors = int(df.get("block_errors", pd.Series(dtype=float)).fillna(0).sum())
        num_blocks = int(df.get("num_blocks", pd.Series(dtype=float)).fillna(0).sum())
        num_batches = int(df.get("num_batches_run", pd.Series(dtype=float)).fillna(0).sum())

    target_met = (num_batches >= min_batches) and (block_errors >= target)
    max_met = num_batches >= max_batches or len(chunks_done) >= max_chunks
    done = bool(target_met or max_met)

    next_chunk = None
    if not done:
        for k in range(max_chunks):
            if k not in chunks_done:
                next_chunk = k
                break
        if next_chunk is None:
            done = True
            max_met = True

    status = {
        "done": done,
        "reason": "target_block_errors" if target_met else ("max_batches" if max_met else "continue"),
        "next_chunk": -1 if next_chunk is None else int(next_chunk),
        "chunks_done": sorted(chunks_done),
        "num_chunks_done": len(chunks_done),
        "bit_errors": bit_errors,
        "num_bits": num_bits,
        "block_errors": block_errors,
        "num_blocks": num_blocks,
        "num_batches": num_batches,
        "target_block_errors": target,
        "min_batches": min_batches,
        "max_batches": max_batches,
        "chunk_batches": chunk_batches,
        "max_chunks": max_chunks,
        "ber": float(bit_errors / num_bits) if num_bits else None,
        "bler": float(block_errors / num_blocks) if num_blocks else None,
    }

    if args.shell:
        for k, v in status.items():
            if isinstance(v, bool):
                v = 1 if v else 0
            elif isinstance(v, list):
                v = ",".join(str(x) for x in v)
            print(f"{k.upper()}={v}")
    else:
        print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
