#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPAIR_ROOT = ROOT / "_isolated_eval_chunks"
BASE_ROOT = ROOT / "_final_u3_baseline_chunks"
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
EBNOS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]


def training_state(variant: str) -> tuple[str, int, int, str]:
    p = ROOT / "TWC_plots_comprehensive" / "runs_rx16" / "seed7" / "1dmrs" / variant / "metrics" / "train_state.json"
    if not p.exists():
        return "MISSING", -1, 40000, "no train_state.json"
    try:
        d = json.loads(p.read_text())
        return (
            "COMPLETE" if d.get("training_complete") else "INCOMPLETE",
            int(d.get("latest_step", -1)),
            int(d.get("total_steps", 40000)),
            str(d.get("save_reason", "")),
        )
    except Exception as exc:
        return "INVALID", -1, 40000, str(exc)


def point(root: Path, variant: str, receiver: str, ebno: float) -> tuple[int, int, bool]:
    frames = []
    for p in root.rglob("chunk_result.csv"):
        try:
            d = pd.read_csv(p)
            if d.empty:
                continue
            r = d.iloc[0]
            if (str(r.get("variant", "")) == variant and
                str(r.get("receiver", "")) == receiver and
                int(r.get("num_users", -1)) == 3 and
                abs(float(r.get("ebno_db")) - ebno) < 1e-9):
                frames.append(d)
        except Exception:
            pass
    if not frames:
        return 0, 0, False
    df = pd.concat(frames, ignore_index=True)
    errors = int(pd.to_numeric(df.get("block_errors"), errors="coerce").fillna(0).sum())
    batches = int(pd.to_numeric(df.get("num_batches_run"), errors="coerce").fillna(0).sum())
    done = (batches >= 20 and errors >= 100) or batches >= 2000
    return errors, batches, done


print("TRAINING")
all_training = True
for v in VARIANTS:
    status, step, total, reason = training_state(v)
    all_training &= status == "COMPLETE"
    print(f"{v:27s} {status:10s} {step:5d}/{total:<5d} reason={reason}")

print("\nUPAIR U=3, Eb/N0=-4..+1")
all_upair = True
for v in VARIANTS:
    states = []
    for e in EBNOS:
        err, batches, done = point(UPAIR_ROOT, v, "upair5g_lmmse", e)
        all_upair &= done
        states.append(f"{e:+g}:{'D' if done else 'M'}({err}/{batches})")
    print(f"{v:27s} " + "  ".join(states))

print("\nBASELINES U=3, Eb/N0=-4..+1")
all_base = True
for r in BASELINES:
    states = []
    for e in EBNOS:
        err, batches, done = point(BASE_ROOT, "main_d256_b4_r2", r, e)
        all_base &= done
        states.append(f"{e:+g}:{'D' if done else 'M'}({err}/{batches})")
    print(f"{r:35s} " + "  ".join(states))

print("\nLegend: D=done under 100-errors-or-2000-batches policy; M=missing/incomplete.")
print(f"OVERALL_COMPLETE={int(all_training and all_upair and all_base)}")
