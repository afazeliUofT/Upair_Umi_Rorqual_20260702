#!/usr/bin/env bash
set -euo pipefail

TASKS="eval_main_u3/tasks.tsv"
: > "$TASKS"

for e in -4 -3 -2 -1 0 1; do
    echo -e "upair5g_lmmse\t$e" >> "$TASKS"
done

for e in -4 -3 -2 -1 0 1; do
    echo -e "baseline_ls_lmmse\t$e" >> "$TASKS"
done

for e in -4 -3 -2 -1 0 1; do
    echo -e "baseline_ls_2dlmmse_lmmse\t$e" >> "$TASKS"
done

for e in -4 -3 -2 -1; do
    echo -e "perfect_csi_lmmse\t$e" >> "$TASKS"
done

nl -ba "$TASKS"
