from pathlib import Path
import re
import pandas as pd

ROOT = Path.cwd()
LOG_ROOT = ROOT / "logs" / "umi_training"
OUT = ROOT / "UMI_training" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT_BY_TASK = {
    "0": "main_d256_b4_r2",
    "1": "shallow_d256_b2_r2",
    "2": "deep_d256_b6_r2",
    "3": "narrow_d192_b4_r2",
    "4": "wide_d320_b4_r2",
    "5": "wide_deep_d320_b6_r2",
    "6": "mlpwide_d256_b4_r4",
}

rows = []

for p in sorted(LOG_ROOT.glob("umi-optB*.out")):
    text = p.read_text(errors="ignore").splitlines()

    job = ""
    task = ""
    m = re.search(r"_(\d+)_(\d+)\.out$", p.name)
    if m:
        job, task = m.group(1), m.group(2)

    variant = VARIANT_BY_TASK.get(task, "")
    host = ""
    gpu_name = ""
    gpu_mem_total_mib = None
    mode = "batch32"
    if "optB-lm" in p.name:
        mode = "lowmem_batch16"
    if "optB-ulm" in p.name:
        mode = "ultralow_batch8"

    current_trial = None
    max_current_gib = 0.0
    max_peak_gib = 0.0
    step_count = 0
    last_step = None
    oom_count = 0
    gpucheck_pass = 0
    gpucheck_fail = 0

    def flush():
        nonlocal_vars = None
        if current_trial is None and step_count == 0 and oom_count == 0:
            return
        rows.append({
            "log_file": str(p),
            "job": job,
            "task": task,
            "variant": variant,
            "mode": mode,
            "host": host,
            "gpu_name": gpu_name,
            "gpu_mem_total_mib": gpu_mem_total_mib,
            "trial": current_trial,
            "last_step_seen": last_step,
            "num_step_log_lines": step_count,
            "max_current_gib": max_current_gib,
            "max_peak_gib": max_peak_gib,
            "oom_count": oom_count,
            "gpucheck_pass": gpucheck_pass,
            "gpucheck_fail": gpucheck_fail,
        })

    for line in text:
        mh = re.search(r"\[GPUCHECK\] host=(\S+)", line)
        if mh:
            host = mh.group(1)

        mg = re.search(r"GPU 0:\s*(.*?)\s*\(UUID:", line)
        if mg:
            gpu_name = mg.group(1).strip()

        mt = re.search(r"/\s*(\d+)MiB", line)
        if mt and "MiB" in line and "Memory-Usage" not in line:
            try:
                gpu_mem_total_mib = max(gpu_mem_total_mib or 0, int(mt.group(1)))
            except Exception:
                pass

        if "GPUCHECK] PASS" in line:
            gpucheck_pass += 1

        if "GPUCHECK-FATAL" in line or "tf_gpus = []" in line:
            gpucheck_fail += 1

        mtrial = re.search(r"starting stage=B trial=(\d+)", line)
        if mtrial:
            if current_trial is not None or step_count or oom_count:
                flush()
            current_trial = int(mtrial.group(1))
            max_current_gib = 0.0
            max_peak_gib = 0.0
            step_count = 0
            last_step = None
            oom_count = 0

        mmem = re.search(
            r"step=(\d+).*gpu_mem=([0-9.]+)GiB peak=([0-9.]+)GiB",
            line,
        )
        if mmem:
            last_step = int(mmem.group(1))
            current = float(mmem.group(2))
            peak = float(mmem.group(3))
            max_current_gib = max(max_current_gib, current)
            max_peak_gib = max(max_peak_gib, peak)
            step_count += 1

        if (
            "ResourceExhausted" in line
            or "CUDA_ERROR_OUT_OF_MEMORY" in line
            or "cuMemAllocAsync failed" in line
            or "out of memory" in line.lower()
        ):
            oom_count += 1

    flush()

df = pd.DataFrame(rows)

if df.empty:
    print("No Stage-B memory rows found.")
else:
    df.to_csv(OUT / "stageB_gpu_memory_by_trial.csv", index=False)

    summary = (
        df.groupby(["mode", "variant"], dropna=False)
        .agg(
            trials_seen=("trial", "count"),
            max_peak_gib=("max_peak_gib", "max"),
            max_current_gib=("max_current_gib", "max"),
            max_step_seen=("last_step_seen", "max"),
            oom_events=("oom_count", "sum"),
            gpucheck_pass=("gpucheck_pass", "sum"),
            gpucheck_fail=("gpucheck_fail", "sum"),
        )
        .reset_index()
        .sort_values(["mode", "variant"])
    )

    summary["safe_for_40gb_rule_of_thumb"] = summary["max_peak_gib"].apply(
        lambda x: "likely" if x < 30 else ("fragile" if x < 35 else "risky")
    )

    summary.to_csv(OUT / "stageB_gpu_memory_summary.csv", index=False)

    print("\n=== Stage-B GPU memory summary ===")
    print(summary.to_string(index=False))

    print("\nWrote:")
    print(" ", OUT / "stageB_gpu_memory_by_trial.csv")
    print(" ", OUT / "stageB_gpu_memory_summary.csv")
