Comprehensive 3-user UPAIR results

BLER:
- BLER=0, NaN, and unavailable points are omitted from BLER figures.
- Filled markers have at least 100 block errors.
- Open markers completed at the 2000-batch cap with fewer than 100 errors.
- An x marker denotes a partial point.

Timing:
- Receiver timing uses receiver_elapsed_s only: estimator plus detector/decoder path.
- Data generation, initialization, warm-up, and shared covariance construction are excluded.
- Values describe the recorded H100 software runs; they are not hardware-independent complexity measures.

NMSE:
- Evaluation NMSE was not collected for the BLER-only isolated runs; no evaluation-NMSE figure was generated.
- Training validation NMSE is available and is shown as UPAIR NMSE / LS NMSE.
- Validation sampled users 1-4 and the configured validation SNR grid; it is not a 3-user test-NMSE curve.

Training timing:
- Logged training time is the sum of step_elapsed_s and excludes queueing/downtime between resumed jobs.
- Median step time excludes validation rows and the first 100 warm-up steps.

Generated figures:
- Fig01_bler_all_methods_u3.png
- Fig02_bler_upair_variants_u3.png
- Fig03_bler_main_vs_benchmarks_u3.png
- Fig04_bler_wide_deep_vs_benchmarks_u3.png
- Fig05_bler_gain_over_2dlmmse_u3.png
- Fig06_receiver_latency_u3.png
- Fig07_receiver_latency_vs_ebno_u3.png
- Fig08_bler_latency_tradeoff_m2dB_u3.png
- Fig10_validation_nmse_ratio.png
- Fig11_training_runtime.png
- Fig12_training_step_time.png
- Fig13_model_size_vs_validation_nmse.png
