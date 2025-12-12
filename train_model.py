import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# --- CONFIGURATION ---
SRC_DATA_DIR = Path("macaque-dataset")          # existing dataset (your current one)
DST_DATA_DIR = Path("macaque-dataset-yolo")     # new "fixed" dataset created quickly
MODEL_ARCHITECTURE = "yolov8n.pt"

EPOCHS = 3
BATCH_SIZE = 16
PROJECT_NAME = "macaque_detection_run"


def choose_device():
    """
    Chooses GPU if available, otherwise CPU.
    Ultralytics accepts device=0 for GPU 0, or device='cpu'.
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _convert_label_file(src_lbl: Path, dst_lbl: Path) -> int:
    """
    Convert label file from 6 columns:
      class id x y w h
    to 5 columns:
      class x y w h

    Returns number of valid lines written.
    """
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with src_lbl.open("r") as fin, dst_lbl.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # Expected format in your dataset: 6 columns (class, track_id, x, y, w, h)
            if len(parts) == 6:
                cls = parts[0]
                x, y, w, h = parts[2:6]
                fout.write(f"{cls} {x} {y} {w} {h}\n")
                written += 1

            # If already YOLO format
            elif len(parts) == 5:
                fout.write(line + "\n")
                written += 1

            # Otherwise ignore malformed lines
            else:
                continue

    return written


def _safe_symlink(src: Path, dst: Path):
    """Create symlink if possible; fallback to copy if symlinks aren't permitted."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_split(src_img_dir: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path, tag: str):
    """
    Prepare one split by linking/copying images and converting labels into YOLO 5-column format.
    """
    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print(f"⚠ Skipping '{tag}' (missing folders): {src_img_dir} or {src_lbl_dir}")
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_lbl = 0
    kept_lbl = 0

    for src_lbl in src_lbl_dir.glob("*.txt"):
        stem = src_lbl.stem
        src_img = src_img_dir / f"{stem}.jpg"

        if not src_img.exists():
            continue

        _safe_symlink(src_img, dst_img_dir / src_img.name)

        written = _convert_label_file(src_lbl, dst_lbl_dir / src_lbl.name)
        total_lbl += 1
        if written > 0:
            kept_lbl += 1

    print(f"[{tag}] processed labels: {total_lbl}, non-empty after conversion: {kept_lbl}")


def prepare_fixed_dataset():
    """
    Prepare a YOLO-compatible dataset view without re-downloading:
    - images are symlinked (fast)
    - labels are rewritten to 5-column YOLO format (drop 2nd column)
    - train/val are placed in standard YOLO layout
    - test is also converted and kept at root level (DST_DATA_DIR/test/images, DST_DATA_DIR/test/labels)
    - a new YAML is generated (train/val only)
    """
    if not SRC_DATA_DIR.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_DATA_DIR}")

    # Train/Val (standard YOLO layout)
    for split in ["train", "val"]:
        _prepare_split(
            src_img_dir=SRC_DATA_DIR / "images" / split,
            src_lbl_dir=SRC_DATA_DIR / "labels" / split,
            dst_img_dir=DST_DATA_DIR / "images" / split,
            dst_lbl_dir=DST_DATA_DIR / "labels" / split,
            tag=split,
        )

    # Test (root-level in your dataset structure)
    _prepare_split(
        src_img_dir=SRC_DATA_DIR / "test" / "images",
        src_lbl_dir=SRC_DATA_DIR / "test" / "labels",
        dst_img_dir=DST_DATA_DIR / "test" / "images",
        dst_lbl_dir=DST_DATA_DIR / "test" / "labels",
        tag="test",
    )

    # Remove any old caches in destination (Ultralytics caches label parsing)
    for cache in [
        DST_DATA_DIR / "labels" / "train.cache",
        DST_DATA_DIR / "labels" / "val.cache",
    ]:
        if cache.exists():
            cache.unlink()

    # Write YAML for new dataset (train/val only; test kept separate by design)
    yaml_path = DST_DATA_DIR / "macaque_data.yaml"
    yaml_text = f"""# Macaque Object Detection Configuration for YOLOv8 (fixed labels)
path: ./{DST_DATA_DIR.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['macaque']
"""
    yaml_path.write_text(yaml_text)
    print(f"✅ Wrote fixed dataset YAML: {yaml_path}")

    return yaml_path


def train_yolov8_model():
    print("Preparing YOLO-compatible dataset view (dropping 2nd column IDs in labels)...")
    fixed_yaml = prepare_fixed_dataset()

    print(f"\nLoading YOLOv8 model architecture: {MODEL_ARCHITECTURE}")
    model = YOLO(MODEL_ARCHITECTURE)

    print(f"\nStarting training on: {fixed_yaml}")
    print(f"Training parameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}")

    device = choose_device()
    print(f"Training device: {device}")

    results = model.train(
        data=str(fixed_yaml),
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        device=device  # set to "cpu" if needed
    )

    print("\n✅ Training completed successfully.")
    return results


if __name__ == "__main__":
    try:
        train_yolov8_model()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
