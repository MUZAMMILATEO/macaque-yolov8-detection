# utils/common.py
from pathlib import Path


def choose_device():
    """
    Chooses GPU if available, otherwise CPU.
    Ultralytics accepts device=0 for GPU 0, or device='cpu'.
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        # If torch import fails for any reason, be safe and use CPU
        return "cpu"


def find_latest_run_dir(runs_dir: Path, run_prefix: str) -> Path:
    """
    Finds the latest run directory in runs/detect matching:
      <run_prefix>, <run_prefix>2, <run_prefix>3, ...
    Returns the directory with the largest suffix number.
    The base folder without a number counts as 1.

    Example:
      run_prefix="macaque_detection_run"
      candidates: macaque_detection_run, macaque_detection_run2, macaque_detection_run3
      -> returns macaque_detection_run3
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates: list[tuple[int, Path]] = []

    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue

        name = d.name
        if name == run_prefix:
            candidates.append((1, d))
        elif name.startswith(run_prefix):
            suffix = name[len(run_prefix):]  # "", "2", "3", ...
            if suffix.isdigit():
                candidates.append((int(suffix), d))

    if not candidates:
        raise FileNotFoundError(
            f"No run folders found in {runs_dir} with prefix '{run_prefix}'. "
            f"Expected '{run_prefix}' or '{run_prefix}2', etc."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]
