import re
from pathlib import Path


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
        return 1

    return int(m.group(1))


def write_test_yaml(yolo_view_dir: Path) -> Path:
    """
    Creates a YAML that includes the test split, pointing to:
      <yolo_view_dir>/test/images
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


def write_fallback_yaml_val_is_test(yolo_view_dir: Path) -> Path:
    """
    Fallback for older Ultralytics versions that do not support split='test':
    creates a YAML where val points to test/images.
    """
    fallback_yaml = yolo_view_dir / "macaque_data_val_is_test.yaml"
    fallback_yaml.write_text(
        f"""# Fallback config: val points to test split
path: ./{yolo_view_dir.as_posix()}
train: images/train
val: test/images

nc: 1
names: ['macaque']
"""
    )
    return fallback_yaml
