#!/usr/bin/env bash
set -uo pipefail
cd /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable
echo "[PROBE1] host=$(hostname) date=$(date) git=$(git rev-parse --short HEAD)"; nvidia-smi -L
export UPAIR_DMRS_CASE=1dmrs UPAIR_SEED=7
export UPAIR_OPTUNA_STAGEB_PREFIX="clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB"
export UPAIR_PIPELINE_RECEIVERS="perfect_csi_lmmse,baseline_ls_2dlmmse_lmmse"
export UPAIR_PIPELINE_USERS="3"
export UPAIR_PIPELINE_EBNOS="-1,0"
export UPAIR_PIPELINE_CHUNK_BATCHES=20 UPAIR_PIPELINE_MICRO=8
export UPAIR_PIPELINE_TARGET_BLOCK_ERRORS=100
export UPAIR_PIPELINE_MAX_BATCHES=600          # cap: 115,200 blocks/point
bash upair_variant_pipeline_worker.sh main_d256_b4_r2
echo "===================== [PROBE1 SUMMARY] ====================="
python - <<'PY'
import glob, pandas as pd, collections
agg=collections.defaultdict(lambda:[0,0])
for f in glob.glob('_isolated_eval_chunks/main_d256_b4_r2_u3_*lmmse*/chunk_result.csv'):
    r=pd.read_csv(f).iloc[0]
    if r['receiver'] not in ('perfect_csi_lmmse','baseline_ls_2dlmmse_lmmse'): continue
    k=(r['receiver'],float(r['ebno_db'])); agg[k][0]+=int(r['block_errors']); agg[k][1]+=int(r['num_blocks'])
for (rc,e),(be,bl) in sorted(agg.items()):
    print(f"{rc:28s} ebno={e:+.1f}  blk_err={be:5d}/{bl:7d}  BLER={be/max(bl,1):.4e}")
print("upair5g_lmmse reference: -1dB 1.461e-3 (101/69120), 0dB 2.630e-4 (100/380160)")
PY
echo "===================== [/PROBE1 SUMMARY] ===================="
