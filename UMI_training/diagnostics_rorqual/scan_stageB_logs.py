from pathlib import Path
import re
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "UMI_training" / "diagnostics_rorqual"
LOG_ROOT = ROOT / "logs" / "umi_training"

patterns = {
    "gpucheck_pass": r"GPUCHECK\] PASS",
    "gpucheck_fail": r"GPUCHECK-FATAL|tf_gpus = \[\]|CUDA_ERROR_NOT_INITIALIZED",
    "cuda_context": r"cudaSetDevice|context is destroyed|present state",
    "oom": r"ResourceExhausted|CUDA_ERROR_OUT_OF_MEMORY|cuMemAllocAsync failed|out of memory",
    "env_missing": r"ModuleNotFoundError|No module named",
    "failed_trial": r"FAILED trial|failed trial",
    "complete_trial": r"completed trial|complete trial",
}

rows = []
for path in sorted(LOG_ROOT.glob("umi-optB*.out")):
    text = path.read_text(errors="ignore")
    task = ""
    m = re.search(r"_(\d+)_(\d+)\.out$", path.name)
    if m:
        job_id, task = m.group(1), m.group(2)
    else:
        job_id = ""
    host = ""
    mh = re.search(r"\[GPUCHECK\] host=([^\s]+)", text)
    if mh:
        host = mh.group(1)
    rows.append({
        "log": str(path),
        "job_id": job_id,
        "task": task,
        "host": host,
        **{k: len(re.findall(v, text, flags=re.I)) for k, v in patterns.items()},
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "stageB_log_scan.csv", index=False)

print("\n========== LOG SCAN ==========")
if df.empty:
    print("No umi-optB logs found.")
else:
    show = df[
        (df["gpucheck_fail"] > 0)
        | (df["cuda_context"] > 0)
        | (df["oom"] > 0)
        | (df["env_missing"] > 0)
        | (df["failed_trial"] > 0)
    ].copy()
    if show.empty:
        print("No suspicious Stage-B log patterns found.")
    else:
        print(show.to_string(index=False))
