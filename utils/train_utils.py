import os
import shutil
from pathlib import Path


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

            # Expected format: 6 columns (class, track_id, x, y, w, h)
            if len(parts) == 6:
                cls = parts[0]
                x, y, w, h = parts[2:6]
                fout.write(f"{cls} {x} {y} {w} {h}\n")
                written += 1

            # Already YOLO format
            elif len(parts) == 5:
                fout.write(line + "\n")
                written += 1

            # Malformed
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


def _prepare_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    tag: str,
):
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


def prepare_fixed_dataset(
    src_data_dir: Path,
    dst_data_dir: Path,
) -> Path:
    """
    Prepare a YOLO-compatible dataset view without re-downloading:
    - images are symlinked (fast)
    - labels are rewritten to 5-column YOLO format (drop 2nd column)
    - train/val are placed in standard YOLO layout
    - test is also converted and kept at root level (dst_data_dir/test/images, dst_data_dir/test/labels)
    - a new YAML is generated (train/val only)
    """
    if not src_data_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_data_dir}")

    # Train/Val (standard YOLO layout)
    for split in ["train", "val"]:
        _prepare_split(
            src_img_dir=src_data_dir / "images" / split,
            src_lbl_dir=src_data_dir / "labels" / split,
            dst_img_dir=dst_data_dir / "images" / split,
            dst_lbl_dir=dst_data_dir / "labels" / split,
            tag=split,
        )

    # Test (root-level in your dataset structure)
    _prepare_split(
        src_img_dir=src_data_dir / "test" / "images",
        src_lbl_dir=src_data_dir / "test" / "labels",
        dst_img_dir=dst_data_dir / "test" / "images",
        dst_lbl_dir=dst_data_dir / "test" / "labels",
        tag="test",
    )

    # Remove any old caches in destination (Ultralytics caches label parsing)
    for cache in [
        dst_data_dir / "labels" / "train.cache",
        dst_data_dir / "labels" / "val.cache",
    ]:
        if cache.exists():
            cache.unlink()

    # Write YAML for train/val only (test kept separate by design)
    yaml_path = dst_data_dir / "macaque_data.yaml"
    yaml_text = f"""# Macaque Object Detection Configuration for YOLOv8 (fixed labels)
path: ./{dst_data_dir.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['macaque']
"""
    yaml_path.write_text(yaml_text)
    print(f"✅ Wrote fixed dataset YAML: {yaml_path}")

    return yaml_path
