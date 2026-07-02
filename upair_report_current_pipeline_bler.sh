#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
import re
import pandas as pd

print("="*100)
print("[A] Merged isolated-eval CSVs")
print("="*100)

merged_paths = sorted(Path("_isolated_eval_chunks").glob("merged_*.csv"))
if not merged_paths:
    print("[INFO] No merged isolated-eval CSVs found under _isolated_eval_chunks/")
else:
    frames = []
    for p in merged_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] could not read {p}: {e}")
            continue
        df["source"] = str(p)
        frames.append(df)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        cols = [
            "variant", "receiver", "num_users", "ebno_db",
            "bler", "block_errors", "num_blocks", "num_batches_run",
            "ber", "bit_errors", "num_bits", "reliable_bler", "source"
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].sort_values(["variant","num_users","ebno_db","receiver"]).to_string(index=False))

print()
print("="*100)
print("[B] Raw receiver result lines in logs/pipeline")
print("="*100)

pat = re.compile(
    r"\[EVAL\] receiver=\s*(?P<receiver>\S+)\s+"
    r"Eb/N0=\s*(?P<ebno>[-+0-9.]+)\s*dB\s+"
    r"BER=(?P<ber>[0-9.eE+-]+)\s+BLER=(?P<bler>[0-9.eE+-]+).*?"
    r"blk_err=\s*(?P<blk_err>\d+)/(?P<num_blocks>\d+).*?"
    r"batches=\s*(?P<batches>\d+)"
)

rows = []
for p in sorted(Path("logs/pipeline").glob("*.out")):
    text = p.read_text(errors="ignore")
    variant = p.name.replace("pipeline_","").split("_")[0]
    # Better variant extraction from filename:
    mvar = re.match(r"pipeline_(.*)_\d+\.out", p.name)
    if mvar:
        variant = mvar.group(1)
    for m in pat.finditer(text):
        rows.append({
            "source_log": str(p),
            "variant_from_logname": variant,
            "receiver": m.group("receiver"),
            "ebno_db": float(m.group("ebno")),
            "bler": float(m.group("bler")),
            "ber": float(m.group("ber")),
            "block_errors": int(m.group("blk_err")),
            "num_blocks": int(m.group("num_blocks")),
            "num_batches_run": int(m.group("batches")),
        })

if not rows:
    print("[INFO] No '[EVAL] receiver=' result lines found in logs/pipeline/*.out")
else:
    df = pd.DataFrame(rows)
    print(df.sort_values(["variant_from_logname","ebno_db","receiver","source_log"]).to_string(index=False))

print()
print("="*100)
print("[C] Best currently available row per variant/user/EbN0/receiver")
print("="*100)

all_frames = []
if merged_paths:
    for p in merged_paths:
        try:
            df = pd.read_csv(p)
            df["source_type"] = "merged_csv"
            df["source"] = str(p)
            all_frames.append(df)
        except Exception:
            pass
if rows:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"variant_from_logname":"variant"})
    df["num_users"] = pd.NA
    df["source_type"] = "pipeline_log"
    df["source"] = df["source_log"]
    all_frames.append(df)

if not all_frames:
    print("[INFO] Nothing to summarize.")
else:
    df = pd.concat(all_frames, ignore_index=True, sort=False)
    for c in ["block_errors","num_blocks","num_batches_run"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Prefer merged CSV rows over raw logs for same point, and prefer larger num_blocks.
    df["source_rank"] = df["source_type"].map({"merged_csv":0, "pipeline_log":1}).fillna(9)
    df = df.sort_values(["variant","num_users","ebno_db","receiver","source_rank","num_blocks"])
    print(df[[
        c for c in ["variant","num_users","ebno_db","receiver","bler","block_errors","num_blocks","num_batches_run","source_type","source"]
        if c in df.columns
    ]].to_string(index=False))
PY
