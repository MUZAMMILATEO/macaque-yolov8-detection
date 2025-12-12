import sys
from pathlib import Path

from ultralytics import YOLO


# --- CONFIGURATION ---

# Folder prefix created by train_model.py (e.g., macaque_detection_run, macaque_detection_run2, ...)
RUN_PREFIX = "macaque_detection_run"

# Root folder where Ultralytics stores training runs
RUNS_DIR = Path("runs") / "detect"

# Folder containing sample images for inference
SAMPLE_DIR = Path("sample-image")

# Default sample image (must exist in SAMPLE_DIR)
DEFAULT_SAMPLE_NAME = "sample.jpg"

# Output directory for inference results
OUTPUT_DIR = Path("inference_results")


def find_latest_run_dir(runs_dir: Path, run_prefix: str) -> Path:
    """
    Finds the latest run directory in runs/detect matching:
      macaque_detection_run, macaque_detection_run2, macaque_detection_run3, ...
    Returns the directory with the largest suffix number (or 1 for the base folder without a number).
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue

        name = d.name
        if name == run_prefix:
            candidates.append((1, d))  # treat base as suffix=1
        elif name.startswith(run_prefix):
            suffix = name[len(run_prefix):]  # "", "2", "3", ...
            if suffix.isdigit():
                candidates.append((int(suffix), d))

    if not candidates:
        raise FileNotFoundError(
            f"No run folders found in {runs_dir} with prefix '{run_prefix}'. "
            f"Expected something like '{run_prefix}' or '{run_prefix}2'."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def select_source_image(sample_dir: Path, default_name: str) -> Path:
    """
    Selects the image to run inference on:
    - If user provides an argument:
        * If it includes an extension (.jpg/.jpeg/.png), use it directly.
        * If it has no extension, try .jpg, .jpeg, .png in that order.
      If not found: warn and fall back to default.
    - If no argument provided: fall back to default.
    """
    default_path = sample_dir / default_name

    # Ensure sample directory exists
    sample_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = [".jpg", ".jpeg", ".png"]

    # If user provided a filename
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        user_input = sys.argv[1].strip()
        user_path = sample_dir / user_input

        # Case A: user already provided an extension
        if user_path.suffix.lower() in allowed_exts:
            if user_path.exists() and user_path.is_file():
                return user_path
            else:
                print(f"⚠ WARNING: '{user_input}' not found in '{sample_dir}'. Falling back to '{default_name}'.")

        # Case B: user provided no extension -> try allowed extensions
        else:
            for ext in allowed_exts:
                candidate = sample_dir / f"{user_input}{ext}"
                if candidate.exists() and candidate.is_file():
                    return candidate
            print(
                f"⚠ WARNING: '{user_input}' not found in '{sample_dir}' "
                f"with extensions {allowed_exts}. Falling back to '{default_name}'."
            )

    # Fallback
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default sample image not found: {default_path}\n"
            f"Please place a valid image at '{default_path}'."
        )

    return default_path


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
