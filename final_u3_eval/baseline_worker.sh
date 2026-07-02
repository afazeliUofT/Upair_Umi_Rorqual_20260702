#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="$ROOT/configs/twc_comprehensive_mu32_base.yaml"
VARIANT="main_d256_b4_r2"
PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
OUT_ROOT="$ROOT/_final_u3_baseline_chunks"
SHARED="$ROOT/_final_u3_shared_cov/u3_prb8_cdlC_covariance/artifacts/empirical_covariances.npz"
CHUNK_BATCHES=20
MICRO=8
TARGET=100
MAX_BATCHES=2000
MIN_BATCHES=20

python "$SCRIPT_DIR/build_shared_covariance.py"
[[ -s "$SHARED" ]] || { echo "[FATAL] Missing shared covariance: $SHARED" >&2; exit 3; }
mkdir -p "$OUT_ROOT"

receivers=(baseline_ls_lmmse baseline_ls_2dlmmse_lmmse perfect_csi_lmmse)
ebnos=(-4 -3 -2 -1 0 1)

for receiver in "${receivers[@]}"; do
  for ebno in "${ebnos[@]}"; do
    echo "================================================================================"
    echo "[FINAL-BASELINE] receiver=$receiver U=3 Eb/N0=$ebno"
    while true; do
      status_file="$(mktemp)"
      python "$ROOT/scripts/isolated_eval_status.py" \
        --input-root "$OUT_ROOT" \
        --config "$CONFIG" \
        --variant "$VARIANT" \
        --receiver "$receiver" \
        --num-users 3 \
        --ebno-db "$ebno" \
        --chunk-batches "$CHUNK_BATCHES" \
        --target-block-errors "$TARGET" \
        --max-batches "$MAX_BATCHES" \
        --min-batches "$MIN_BATCHES" \
        --shell > "$status_file"
      # shellcheck disable=SC1090
      source "$status_file"
      rm -f "$status_file"
      echo "[FINAL-BASELINE] done=$DONE reason=$REASON batches=$NUM_BATCHES errors=$BLOCK_ERRORS next=$NEXT_CHUNK"
      [[ "$DONE" == "1" ]] && break

      common=(
        --config "$CONFIG"
        --variant "$VARIANT"
        --dmrs-case 1dmrs
        --seed 7
        --num-users 3
        --receiver "$receiver"
        --ebno-db "$ebno"
        --chunk-idx "$NEXT_CHUNK"
        --chunk-batches "$CHUNK_BATCHES"
        --receiver-microbatch-size "$MICRO"
        --stageb-prefix "$PREFIX"
        --optuna-dir "$ROOT/optuna"
        --output-root "$OUT_ROOT"
      )

      if [[ "$receiver" == "baseline_ls_2dlmmse_lmmse" ]]; then
        python -u "$SCRIPT_DIR/run_chunk_with_shared_cov.py" \
          "${common[@]}" --shared-cov-cache "$SHARED"
      else
        python -u "$ROOT/scripts/run_isolated_eval_chunk.py" "${common[@]}"
      fi
    done

    safe="${ebno//-/m}"
    safe="${safe//./p}"
    python "$ROOT/scripts/merge_isolated_eval_chunks.py" \
      --input-root "$OUT_ROOT" \
      --output-csv "$OUT_ROOT/merged_${VARIANT}_u3_${receiver}_e${safe}.csv" \
      --variant "$VARIANT" \
      --receiver "$receiver" \
      --num-users 3 \
      --ebno-db "$ebno"
  done
done

echo "[FINAL-BASELINE] COMPLETE"
