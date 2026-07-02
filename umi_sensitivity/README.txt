UMi sensitivity profile
=======================
Evaluation only. No training function is called.

Channel:
- Sionna UMi, uplink, one BS, three UTs.
- Fresh single-sector topology for every channel call.
- Standard UMi geometry: 10 m minimum BS-UT distance, 200 m ISD,
  10 m BS height, 1.5 m UT height, 0.8 indoor probability.
- Mobility 8.33--16.67 m/s, matching CDL-C training.
- Same one-antenna UT and 1x16 omni BS array as training.
- LoS/NLoS sampled according to UMi; not forced.
- Pathloss and shadow fading disabled.
- Per-link resource-grid normalization enabled.

Interpretation:
This is a normalized small-scale channel-mismatch test. It is not a
link-budget, power-control, near-far, pathloss, or shadow-fading test.

Evaluation:
- U=3
- Eb/N0=-4,-3,-2,-1,0,+1 dB
- 100 block errors or 2000 logical batches
- 20 batches/chunk
- receiver microbatch 8
- seven UPAIR jobs plus one benchmark job

Separate outputs:
- _umi_eval_chunks
- _umi_baseline_chunks
- _umi_shared_cov
- logs/umi_sensitivity

The seven trained checkpoints are protected by a SHA-256 manifest.
