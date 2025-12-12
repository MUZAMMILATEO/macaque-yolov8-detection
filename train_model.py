from pathlib import Path
from ultralytics import YOLO
from utils.commons import choose_device
from utils.train_utils import prepare_fixed_dataset


# --- CONFIGURATION ---
SRC_DATA_DIR = Path("macaque-dataset")          # existing dataset (your current one)
DST_DATA_DIR = Path("macaque-dataset-yolo")     # new "fixed" dataset created quickly
MODEL_ARCHITECTURE = "yolov8n.pt"

EPOCHS = 20
BATCH_SIZE = 16
PROJECT_NAME = "macaque_detection_run"


def train_yolov8_model():
    print("Preparing YOLO-compatible dataset view (dropping 2nd column IDs in labels)...")
    fixed_yaml = prepare_fixed_dataset(SRC_DATA_DIR, DST_DATA_DIR)

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
        device=device,
    )

    print("\n✅ Training completed successfully.")
    return results


if __name__ == "__main__":
    try:
        train_yolov8_model()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
