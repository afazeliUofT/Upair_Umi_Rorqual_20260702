#!/usr/bin/env bash
set -euo pipefail

domain="${1:?Usage: eval_variant_worker.sh <umi|cdlc> <variant>}"
variant="${2:?Usage: eval_variant_worker.sh <umi|cdlc> <variant>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

prefix="umiNorm_v1_b32_prb8_u34610_1dmrs_stageB"
checkpoint="$ROOT/UMI_training/runs_rx16/seed7/1dmrs/$variant/checkpoints/best.weights.h5"

case "$domain" in
  umi)
    config="$ROOT/configs/twc_comprehensive_mu32_umi_training.yaml"
    output="$ROOT/_umi_trained_umi_eval_chunks"
    ;;
  cdlc)
    config="$ROOT/configs/twc_comprehensive_mu32_base.yaml"
    output="$ROOT/_umi_trained_cdlc_eval_chunks"
    ;;
  *)
    echo "[FATAL] domain must be umi or cdlc" >&2
    exit 2
    ;;
esac

[[ -s "$checkpoint" ]] || { echo "[FATAL] Missing checkpoint: $checkpoint" >&2; exit 3; }
mkdir -p "$output"

for ebno in -4 -3 -2 -1 0 1; do
  while true; do
    status="$(mktemp)"
    python scripts/isolated_eval_status.py \
      --input-root "$output" \
      --config "$config" \
      --variant "$variant" \
      --receiver upair5g_lmmse \
      --num-users 3 \
      --ebno-db "$ebno" \
      --chunk-batches 20 \
      --target-block-errors 100 \
      --max-batches 2000 \
      --min-batches 20 \
      --shell > "$status"
    # shellcheck disable=SC1090
    source "$status"
    rm -f "$status"

    echo "[$domain] variant=$variant ebno=$ebno done=$DONE batches=$NUM_BATCHES errors=$BLOCK_ERRORS next=$NEXT_CHUNK"
    [[ "$DONE" == "1" ]] && break

    python -u scripts/run_isolated_eval_chunk.py \
      --config "$config" \
      --variant "$variant" \
      --dmrs-case 1dmrs \
      --seed 7 \
      --num-users 3 \
      --receiver upair5g_lmmse \
      --ebno-db "$ebno" \
      --chunk-idx "$NEXT_CHUNK" \
      --chunk-batches 20 \
      --receiver-microbatch-size 8 \
      --stageb-prefix "$prefix" \
      --optuna-dir "$ROOT/optuna" \
      --output-root "$output" \
      --checkpoint "$checkpoint"
  done

  safe="${ebno//-/m}"
  safe="${safe//./p}"
  python scripts/merge_isolated_eval_chunks.py \
    --input-root "$output" \
    --output-csv "$output/merged_${variant}_u3_upair5g_lmmse_e${safe}.csv" \
    --variant "$variant" \
    --receiver upair5g_lmmse \
    --num-users 3 \
    --ebno-db "$ebno"
done
