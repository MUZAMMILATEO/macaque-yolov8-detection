import sys
from pathlib import Path
from ultralytics import YOLO

from utils.commons import choose_device, find_latest_run_dir

from utils.infer_utils import select_source_image


# --- CONFIGURATION ---

# Folder prefix created by train_model.py (e.g., macaque_detection_run, macaque_detection_run2, ...)
RUN_PREFIX = "macaque_detection_run"

# Root folder where Ultralytics stores training runs
RUNS_DIR = Path("runs") / "detect"

# Folder containing sample images for inference
SAMPLE_DIR = Path("sample-image")  # keep your folder name as-is

# Default sample image (must exist in SAMPLE_DIR)
DEFAULT_SAMPLE_NAME = "sample.jpg"

# Output directory for inference results
OUTPUT_DIR = Path("inference_results")


def run_inference(model_path: Path, source_path: Path, output_dir: Path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model weights not found at: {model_path}")

    print(f"Loading trained model from: {model_path}")
    model = YOLO(str(model_path))

    print(f"Running inference on: {source_path}")
    print(f"Device: {device}")

    results = model.predict(
        source=str(source_path),
        conf=0.25,
        iou=0.7,
        save=True,
        project=str(output_dir.parent),
        name=str(output_dir.name),
        exist_ok=True,
        device=device,
    )

    print(f"\n✅ Inference completed. Results saved under: {output_dir.resolve()}")

    # Print quick stats for the first result
    if results:
        r0 = results[0]
        n = len(r0.boxes) if getattr(r0, "boxes", None) is not None else 0
        print(f"   -> Detected {n} macaque(s) in {source_path.name}")


if __name__ == "__main__":
    try:
        # 1) Find latest training run
        latest_run_dir = find_latest_run_dir(RUNS_DIR, RUN_PREFIX)
        weights_path = latest_run_dir / "weights" / "best.pt"

        # 2) Choose source image (user input filename or fallback)
        source_img = select_source_image(SAMPLE_DIR, DEFAULT_SAMPLE_NAME)

        # 3) Choose device
        device = choose_device()

        # 4) Run inference
        run_inference(weights_path, source_img, OUTPUT_DIR, device)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nUsage:")
        print("  python infer_model.py <image_filename_in_sample_image_folder>")
        print("Examples:")
        print("  python infer_model.py my_test.jpg")
        print("  python infer_model.py my_test.png")
        print("  python infer_model.py my_test      # tries .jpg/.jpeg/.png")
        print(f"\nIf no filename is provided, it falls back to: {SAMPLE_DIR / DEFAULT_SAMPLE_NAME}")
        sys.exit(1)
