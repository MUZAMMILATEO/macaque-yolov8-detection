import sys
from pathlib import Path
from ultralytics import YOLO

from utils.commons import choose_device, find_latest_run_dir

from utils.eval_utils import (
    extract_run_suffix,
    write_fallback_yaml_val_is_test,
    write_test_yaml,
)

# --- CONFIGURATION ---
RUN_PREFIX = "macaque_detection_run"
RUNS_DIR = Path("runs") / "detect"

YOLO_VIEW_DIR = Path("macaque-dataset-yolo")

# Evaluation output base folder:
# runs/detect/macaque_test_evaluation/<N>  (N matches the training run suffix)
EVAL_PROJECT_DIR = Path("runs") / "detect" / "macaque_test_evaluation"


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

    # Save to: runs/detect/macaque_test_evaluation/<run_num>/
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
        fallback_yaml = write_fallback_yaml_val_is_test(YOLO_VIEW_DIR)
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
