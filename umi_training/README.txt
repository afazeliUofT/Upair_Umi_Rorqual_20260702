UMi training workflow
=====================

Primary model:
  Train from scratch on normalized Sionna UMi.
  Seven architecture variants run as seven independent one-GPU Slurm tasks.

Tuning:
  Stage A: 20 trials x 4,000 steps per variant.
           The first queued candidate is that variant's CDL-C Stage-B best.
  Stage B: re-run the top 6 Stage-A candidates x 12,000 steps.
  Stage C: not used.
  Final:   one fresh 40,000-step training run using the UMi Stage-B winner.

Why:
  The channel distribution has changed. UMi-specific hyperparameter tuning is
  required for a primary UMi result, while Stage C would duplicate much of the
  final 40,000-step training cost.

Outputs:
  UMI_training/optuna_db/
  UMI_training/runs_rx16/seed7/1dmrs/
  _umi_trained_umi_eval_chunks/
  _umi_trained_cdlc_eval_chunks/

Existing CDL-C checkpoints remain protected by umi_sensitivity's SHA-256 guard.

Sequence:
  bash umi_training/submit_smoke.sh
  python umi_training/driver.py smoke-status

  bash umi_training/submit_stageA.sh
  python umi_training/driver.py study-status A
  # Resubmit submit_stageA.sh until STAGE_A_COMPLETE=1.

  bash umi_training/submit_stageB.sh
  python umi_training/driver.py study-status B
  # Resubmit submit_stageB.sh until STAGE_B_COMPLETE=1.

  bash umi_training/submit_final_training.sh
  python umi_training/driver.py training-status
  # Resubmit until UMI_FINAL_TRAINING_COMPLETE=1.

  bash umi_training/submit_eval_umi.sh
  python umi_training/driver.py eval-status umi

  bash umi_training/submit_eval_cdlc.sh
  python umi_training/driver.py eval-status cdlc

The existing UMi and CDL-C benchmark curves are independent of neural-network
training and can be reused. Do not recompute them unless their status is
incomplete.
