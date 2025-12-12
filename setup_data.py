import os
import shutil
import zipfile
import tarfile
import requests
import random
import sys

# --- CONFIGURATION ---

# The most likely working direct API URL for the data file's content
DIRECT_DOWNLOAD_URL = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId/?persistentId=doi:10.25625/CMQY0Q/ZH2YKH"

# Name of the output folder for the final YOLO structure
YOLO_DATA_DIR = "macaque-dataset"

# Name of the archive file that is expected to be downloaded or manually placed
# We use .zip as a placeholder name, but the content is likely a .tar.gz
ZIP_FILENAME = "macaca_data_specific.zip"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def download_data():
    """
    Downloads the dataset. Prioritizes finding a manually placed file.
    If not found, it attempts the direct API download.
    """

    # 1. Check for manual download fallback
    if os.path.exists(ZIP_FILENAME):
        print(f"✅ Found existing file '{ZIP_FILENAME}'. Skipping download.")
        return True

    # 2. Attempt direct API download
    print(f"File not found. Attempting to download specific file from: {DIRECT_DOWNLOAD_URL}")

    try:
        # Increase timeout as this is a large file
        response = requests.get(DIRECT_DOWNLOAD_URL, stream=True, timeout=600)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192  # 8 KiB

        with open(ZIP_FILENAME, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print download progress
                    if total_size > 0:
                        status = (
                            f"Downloaded {downloaded / (1024*1024):.2f} MB of "
                            f"{total_size / (1024*1024):.2f} MB"
                        )
                    else:
                        status = f"Downloaded {downloaded / (1024*1024):.2f} MB"
                    sys.stdout.write(f"\r{status}")
                    sys.stdout.flush()

        print(f"\n✅ Successfully downloaded {ZIP_FILENAME} (Size: {downloaded / (1024*1024*1024):.2f} GB)")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error during download: {e}")
        print("\n*** ACTION REQUIRED ***")
        print(f"Please manually download the file using the link below and rename it to '{ZIP_FILENAME}'.")
        print("Manual Link: https://data.goettingen-research-online.de/file.xhtml?persistentId=doi:10.25625/CMQY0Q/ZH2YKH&version=1.0")
        print("***********************")
        return False


def extract_and_organize(archive_file, target_dir):
    """
    Extracts the archive using zipfile, then tarfile as a fallback,
    and organizes the files into the YOLO structure.

    NEW: also creates a TEST split at:
      <target_dir>/test/images
      <target_dir>/test/labels
    """

    print(f"\nExtracting {archive_file}...")
    temp_extract_dir = "temp_extracted_data"
    os.makedirs(temp_extract_dir, exist_ok=True)

    extraction_successful = False

    # 1. Attempt Extraction with ZIP
    try:
        with zipfile.ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)
            print("   -> Extracted successfully using zipfile.")
            extraction_successful = True
    except zipfile.BadZipFile:
        print("   -> zipfile failed. Trying tarfile...")

    # 2. Attempt Extraction with TAR/GZ (Fallback)
    if not extraction_successful:
        try:
            with tarfile.open(archive_file, "r:*") as tar_ref:
                tar_ref.extractall(temp_extract_dir)
                print("✅ Extracted successfully using tarfile (likely .tar.gz).")
                extraction_successful = True
        except tarfile.TarError as e:
            print(f"❌ Both zipfile and tarfile failed. Error: {e}")
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            return

    # 3. Handle failure
    if not extraction_successful:
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return

    # 4. Find the root data directory after extraction
    extracted_root_contents = os.listdir(temp_extract_dir)
    if not extracted_root_contents:
        print("\n❌ Error: Extracted directory is empty.")
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return

    # Assuming the archive contains a single folder at the top level
    extracted_root = os.path.join(temp_extract_dir, extracted_root_contents[0])

    # 5. Consolidate files
    all_files = []
    print("\nConsolidating files and preparing for split...")

    for i in range(10):
        folder_name = f"{i}_100"
        folder_path = os.path.join(extracted_root, folder_name)

        images_path = os.path.join(folder_path, "images")
        labels_path = os.path.join(folder_path, "labels_with_ids")

        if os.path.isdir(images_path) and os.path.isdir(labels_path):
            image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]

            for img_filename in image_files:
                base_name = os.path.splitext(img_filename)[0]
                label_filename = base_name + ".txt"

                img_src = os.path.join(images_path, img_filename)
                lbl_src = os.path.join(labels_path, label_filename)

                if os.path.exists(lbl_src):
                    all_files.append((img_src, lbl_src, base_name))

    print(f"Found {len(all_files)} matching image/label pairs.")

    if not all_files:
        print("\n❌ Error: No matching image/label pairs found.")
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return

    # 6. Perform Train/Val/Test split
    random.shuffle(all_files)

    total = len(all_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(
        f"Train samples: {len(train_files)}, "
        f"Validation samples: {len(val_files)}, "
        f"Test samples: {len(test_files)}"
    )

    # 7. Create output directory structure
    print("\nCreating final YOLO directory structure...")

    # Standard YOLO train/val layout
    os.makedirs(os.path.join(target_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels", "val"), exist_ok=True)

    # Test set at ROOT level (not under images/ or labels/)
    os.makedirs(os.path.join(target_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test", "labels"), exist_ok=True)

    def copy_files(file_list, split_type):
        for img_src, lbl_src, base_name in file_list:
            if split_type == "test":
                img_dst = os.path.join(target_dir, "test", "images", base_name + ".jpg")
                lbl_dst = os.path.join(target_dir, "test", "labels", base_name + ".txt")
            else:
                img_dst = os.path.join(target_dir, "images", split_type, base_name + ".jpg")
                lbl_dst = os.path.join(target_dir, "labels", split_type, base_name + ".txt")

            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    print(f"✅ Data structured successfully in '{target_dir}'")

    # 8. Cleanup
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    print(f"Cleaned up temporary extraction directory: {temp_extract_dir}")
    os.remove(archive_file)
    print(f"Cleaned up original archive file: {archive_file}")


def create_yaml_config(data_dir):
    """Creates the YOLO configuration file required for training."""
    yaml_content = f"""
# Macaque Object Detection Configuration for YOLOv8
# Path is relative to the directory where the training script is executed
path: ./{data_dir}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names (ID 0 is for 'macaque')
names: ['macaque']
"""
    yaml_path = os.path.join(data_dir, "macaque_data.yaml")

    os.makedirs(data_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ Created YOLO configuration file: '{yaml_path}'")


if __name__ == "__main__":
    # If the file exists, check if it's potentially an oversized, incorrect download (> 5GB)
    if os.path.exists(ZIP_FILENAME) and os.path.getsize(ZIP_FILENAME) > 5 * 1024 * 1024 * 1024:
        print(f"Detected previous, oversized download: {ZIP_FILENAME}. Deleting...")
        os.remove(ZIP_FILENAME)

    if download_data():
        extract_and_organize(ZIP_FILENAME, YOLO_DATA_DIR)
        create_yaml_config(YOLO_DATA_DIR)
        print("\n✨ Data setup complete! Ready for YOLOv8 training.")
    else:
        print("\n🛑 Setup incomplete. Please resolve the download/network issue and try again.")
