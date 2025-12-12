import re
import sys
from pathlib import Path

from ultralytics import YOLO


# --- CONFIGURATION ---
RUN_PREFIX = "macaque_detection_run"
RUNS_DIR = Path("runs") / "detect"

YOLO_VIEW_DIR = Path("macaque-dataset-yolo")

# Evaluation output base folder:
# runs/detect/macaque_test_evaluation/<N>  (N matches the training run suffix)
EVAL_PROJECT_DIR = Path("runs") / "detect" / "macaque_test_evaluation"


def find_latest_run_dir(runs_dir: Path, run_prefix: str) -> Path:
    """
    Finds latest run directory in runs/detect matching:
      macaque_detection_run, macaque_detection_run2, macaque_detection_run3, ...
    Returns the directory with the largest suffix number (base without number counts as 1).
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue

        name = d.name
        if name == run_prefix:
            candidates.append((1, d))
        elif name.startswith(run_prefix):
            suffix = name[len(run_prefix):]
            if suffix.isdigit():
                candidates.append((int(suffix), d))

    if not candidates:
        raise FileNotFoundError(
            f"No run folders found in {runs_dir} with prefix '{run_prefix}'. "
            f"Expected '{run_prefix}' or '{run_prefix}2', etc."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def extract_run_suffix(run_dir: Path, run_prefix: str) -> int:
    """
    Extracts numeric suffix from a run folder name.
    - macaque_detection_run    -> 1
    - macaque_detection_run2   -> 2
    - macaque_detection_run15  -> 15
    """
    name = run_dir.name
    if name == run_prefix:
        return 1
    m = re.match(rf"^{re.escape(run_prefix)}(\d+)$", name)
    if not m:
        # Fallback: if unexpected naming, still produce something stable
        return 1
    return int(m.group(1))


def choose_device():
    """GPU if available else CPU."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def write_test_yaml(yolo_view_dir: Path) -> Path:
    """
    Creates a YAML that includes the test split, pointing to:
      macaque-dataset-yolo/test/images
    This file is created when eval_model.py runs.
    """
    test_images = yolo_view_dir / "test" / "images"
    test_labels = yolo_view_dir / "test" / "labels"
    if not test_images.exists() or not test_labels.exists():
        raise FileNotFoundError(
            "YOLO-view test set not found.\n"
            f"Expected:\n  {test_images}\n  {test_labels}\n\n"
            "Please run `python train_model.py` first so it prepares macaque-dataset-yolo/ (including test)."
        )

    yaml_path = yolo_view_dir / "macaque_data_with_test.yaml"
    yaml_text = f"""# Macaque YOLOv8 dataset config (includes test split)
path: ./{yolo_view_dir.as_posix()}
train: images/train
val: images/val
test: test/images

nc: 1
names: ['macaque']
"""
    yaml_path.write_text(yaml_text)
    return yaml_path


def evaluate_on_test():
    # 1) Find latest trained model
    latest_run_dir = find_latest_run_dir(RUNS_DIR, RUN_PREFIX)
    run_num = extract_run_suffix(latest_run_dir, RUN_PREFIX)

    weights_path = latest_run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found at: {weights_path}")

    # 2) Create YAML that includes test split (generated on the fly)
    test_yaml = write_test_yaml(YOLO_VIEW_DIR)

    # 3) Choose device
    device = choose_device()

    print(f"Using model: {weights_path}")
    print(f"Training run detected: {latest_run_dir.name}  -> evaluation will be saved to: {EVAL_PROJECT_DIR / str(run_num)}")
    print(f"Using data YAML: {test_yaml}")
    print(f"Device: {device}")

    # 4) Evaluate on test split
    model = YOLO(str(weights_path))

    # We save to: runs/detect/macaque_test_evaluation/<run_num>/
    project = str(EVAL_PROJECT_DIR)
    name = str(run_num)

    try:
        metrics = model.val(
            data=str(test_yaml),
            split="test",
            imgsz=640,
            batch=16,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
        )
    except TypeError:
        # Fallback for older ultralytics: val doesn't accept split="test"
        fallback_yaml = YOLO_VIEW_DIR / "macaque_data_val_is_test.yaml"
        fallback_yaml.write_text(
            f"""# Fallback config: val points to test split
path: ./{YOLO_VIEW_DIR.as_posix()}
train: images/train
val: test/images

nc: 1
names: ['macaque']
"""
        )
        print("⚠ Ultralytics version does not support split='test'. Using fallback YAML (val=test) instead.")
        metrics = model.val(
            data=str(fallback_yaml),
            imgsz=640,
            batch=16,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
        )

    print("\n✅ Test evaluation completed.")
    print(f"Results saved under: {(EVAL_PROJECT_DIR / str(run_num)).resolve()}")

    # Compact numeric summary (if available)
    try:
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"mAP50-95:  {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        evaluate_on_test()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nUsage:")
        print("  python eval_model.py")
        sys.exit(1)
