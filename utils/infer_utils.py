import sys
from pathlib import Path


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
