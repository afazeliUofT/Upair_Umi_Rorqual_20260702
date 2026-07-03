#!/usr/bin/env bash
set -euo pipefail

TASK_ID="${1:?task id required}"

ROOT="/home/rsadve1/links/scratch/Extended_UPAIR_Narval_b32m16_portable_underUMI"
cd "$ROOT"

source eval_main_u3/common.sh
activate_env

[[ -s "$CHECKPOINT" ]] || { echo "[FATAL] missing checkpoint: $CHECKPOINT" >&2; exit 3; }

TASK_FILE="$ROOT/eval_main_u3/tasks.tsv"
line="$(sed -n "$((TASK_ID + 1))p" "$TASK_FILE")"
[[ -n "$line" ]] || { echo "[FATAL] no task line for index $TASK_ID" >&2; exit 4; }

receiver="$(echo "$line" | awk '{print $1}')"
ebno="$(echo "$line" | awk '{print $2}')"

mkdir -p "$OUT_ROOT"

echo "============================================================"
echo "[TASK] id=$TASK_ID receiver=$receiver ebno=$ebno"
echo "============================================================"

if [[ "$receiver" == "baseline_ls_2dlmmse_lmmse" ]]; then
    [[ -s "$SHARED_COV" ]] || { echo "[FATAL] missing shared covariance: $SHARED_COV" >&2; exit 5; }
fi

while true; do
    status="$(mktemp)"

    python scripts/isolated_eval_status.py \
      --input-root "$OUT_ROOT" \
      --config "$CONFIG" \
      --variant "$VARIANT" \
      --receiver "$receiver" \
      --num-users 3 \
      --ebno-db "$ebno" \
      --chunk-batches "$CHUNK_BATCHES" \
      --target-block-errors "$TARGET_ERRORS" \
      --max-batches "$MAX_BATCHES" \
      --min-batches "$MIN_BATCHES" \
      --shell > "$status"

    # shellcheck disable=SC1090
    source "$status"
    rm -f "$status"

    echo "[STATUS] receiver=$receiver ebno=$ebno done=$DONE reason=$REASON batches=$NUM_BATCHES errors=$BLOCK_ERRORS next=$NEXT_CHUNK"

    [[ "$DONE" == "1" ]] && break

    if [[ "$receiver" == "baseline_ls_2dlmmse_lmmse" ]]; then
        python -u eval_main_u3/run_chunk_with_shared_cov.py \
          --config "$CONFIG" \
          --variant "$VARIANT" \
          --dmrs-case 1dmrs \
          --seed 7 \
          --num-users 3 \
          --receiver "$receiver" \
          --ebno-db "$ebno" \
          --chunk-idx "$NEXT_CHUNK" \
          --chunk-batches "$CHUNK_BATCHES" \
          --receiver-microbatch-size "$MICRO" \
          --stageb-prefix "$PREFIX" \
          --optuna-dir "$OPTUNA_DIR" \
          --output-root "$OUT_ROOT" \
          --checkpoint "$CHECKPOINT" \
          --shared-cov-cache "$SHARED_COV"
    else
        python -u scripts/run_isolated_eval_chunk.py \
          --config "$CONFIG" \
          --variant "$VARIANT" \
          --dmrs-case 1dmrs \
          --seed 7 \
          --num-users 3 \
          --receiver "$receiver" \
          --ebno-db "$ebno" \
          --chunk-idx "$NEXT_CHUNK" \
          --chunk-batches "$CHUNK_BATCHES" \
          --receiver-microbatch-size "$MICRO" \
          --stageb-prefix "$PREFIX" \
          --optuna-dir "$OPTUNA_DIR" \
          --output-root "$OUT_ROOT" \
          --checkpoint "$CHECKPOINT"
    fi
done

safe="${ebno//-/m}"
safe="${safe//./p}"

python scripts/merge_isolated_eval_chunks.py \
  --input-root "$OUT_ROOT" \
  --output-csv "$OUT_ROOT/merged_${VARIANT}_u3_${receiver}_e${safe}.csv" \
  --variant "$VARIANT" \
  --receiver "$receiver" \
  --num-users 3 \
  --ebno-db "$ebno"

echo "[TASK] DONE id=$TASK_ID receiver=$receiver ebno=$ebno"
