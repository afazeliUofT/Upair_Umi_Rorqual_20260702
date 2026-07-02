from __future__ import annotations

"""Compatibility entry point.

The old in-process Optuna driver was intentionally replaced because running many
TensorFlow trials inside one Python process can poison the CUDA allocator state.
This wrapper always dispatches to the isolated controller, which launches one
fresh TensorFlow worker process per Optuna trial.
"""

from run_optuna_1dmrs_structure_isolated import main


if __name__ == "__main__":
    main()
