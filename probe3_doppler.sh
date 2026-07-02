#!/usr/bin/env bash
set -uo pipefail
cd /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable
source upair_portable_env.sh && upair_activate
for SPD in 8.33 16.67; do
  T=${SPD/./p}
  python - "$SPD" <<'PY'
import sys,yaml
c=yaml.safe_load(open('configs/twc_comprehensive_mu32_base.yaml'))
c['channel']['min_speed_mps']=float(sys.argv[1]); c['channel']['max_speed_mps']=float(sys.argv[1])
yaml.safe_dump(c,open(f"configs/probe3_speed_{sys.argv[1].replace('.','p')}.yaml",'w'))
PY
  for i in $(seq 0 9); do
    python -u scripts/run_isolated_eval_chunk.py \
      --config "configs/probe3_speed_${T}.yaml" \
      --variant main_d256_b4_r2 --dmrs-case 1dmrs --seed 7 \
      --num-users 3 --receiver upair5g_lmmse --ebno-db -1 \
      --chunk-idx $((9100+i)) --chunk-batches 20 --receiver-microbatch-size 8 \
      --stageb-prefix clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB \
      --optuna-dir "$PWD/optuna" --output-root "$PWD/_probe3_speed_${T}" \
      | grep -E "ISO-CHUNK|EVAL\] receiver="
  done
done
echo "===================== [PROBE3 SUMMARY] ====================="
python - <<'PY'
import glob, pandas as pd
for spd in ('8p33','16p67'):
    be=bl=0
    for f in glob.glob(f'_probe3_speed_{spd}/*/chunk_result.csv'):
        r=pd.read_csv(f).iloc[0]; be+=int(r['block_errors']); bl+=int(r['num_blocks'])
    print(f"speed={spd.replace('p','.')} m/s  u3 ebno=-1dB  blk_err={be}/{bl}  BLER={be/max(bl,1):.4e}")
PY
echo "===================== [/PROBE3 SUMMARY] ===================="
