"""
Dataset Download Module for DentalVision-QA
Handles downloading and validation of dental image datasets.
Supports: Zenodo, Kaggle, Roboflow
"""

import os
import sys
import json
import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime

try:
    import cv2
except ImportError:
    print("Warning: OpenCV not installed. Installing...")
    os.system("pip install opencv-python")
    import cv2

try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not installed. Installing...")
    os.system("pip install numpy")
    import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# ─── Helper Functions ──────────────────────────────────────────────────


def create_dummy_dataset(num_images=20):
    """Create placeholder images and YOLO labels for pipeline testing."""
    print(f"[INFO] Creating {num_images} dummy placeholder images...")

    # Create directory structure
    for split in ["train", "val", "test"]:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Dental finding classes
    classes = [
        "caries",
        "plaque",
        "calculus",
        "gingivitis",
        "missing_tooth",
        "filling_crown",
        "ambiguous",
    ]

    # Split: 70/20/10
    splits = {"train": 0.7, "val": 0.2, "test": 0.1}

    created = {"images": 0, "labels": 0}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for i in range(num_images):
        # Determine split
        r = random.random()
        if r < splits["train"]:
            split = "train"
        elif r < splits["train"] + splits["val"]:
            split = "val"
        else:
            split = "test"

        # Create a synthetic dental-like image (grayscale-ish tooth region)
        img_h, img_w = 640, 480
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240  # light background

        # Add some random "tooth" regions (ellipses)
        num_teeth = random.randint(2, 5)
        for _ in range(num_teeth):
            cx = random.randint(80, img_w - 80)
            cy = random.randint(100, img_h - 100)
            rx = random.randint(20, 50)
            ry = random.randint(30, 70)
            color = random.randint(180, 230)
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (color, color, color), -1)

        # Add some noise
        noise = np.random.randint(0, 20, (img_h, img_w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

        # Save image
        img_name = f"dental_{i:04d}.jpg"
        img_path = DATA_DIR / "images" / split / img_name
        cv2.imwrite(str(img_path), img)
        created["images"] += 1

        # Create YOLO label file
        label_path = DATA_DIR / "labels" / split / f"dental_{i:04d}.txt"
        num_objects = random.randint(1, 3)
        with open(label_path, "w") as f:
            for _ in range(num_objects):
                cls = random.randint(0, len(classes) - 1)
                # Random bounding box (normalized)
                bw = random.uniform(0.08, 0.25)
                bh = random.uniform(0.10, 0.35)
                bx = random.uniform(bw / 2, 1 - bw / 2)
                by = random.uniform(bh / 2, 1 - bh / 2)
                f.write(f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
                created["labels"] += 1

        split_counts[split] += 1

    # Create data.yaml
    yaml_content = f"""# DentalVision-QA Dataset Configuration
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

path: {DATA_DIR.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {classes}
"""
    yaml_path = DATA_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] Created data.yaml at {yaml_path}")

    # Create raw placeholder
    (RAW_DIR / "dummy_placeholder").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "dummy_placeholder" / "README.txt").write_text(
        "This dataset was auto-generated as a placeholder because no real dataset was found.\n"
        "To use a real dataset, please download one of the supported datasets:\n"
        "- Zenodo: https://doi.org/10.5281/zenodo.XXXXX (6,313 images)\n"
        "- Kaggle: Dental Cavity Detection Dataset (418 images)\n"
        "- Roboflow: Dental caries dataset\n"
        "Place files in data/raw/ and re-run prepare_dataset.py\n"
    )

    print(f"[INFO] Dummy dataset created successfully!")
    print(
        f"  - Images: {created['images']} (train:{split_counts['train']} val:{split_counts['val']} test:{split_counts['test']})"
    )
    print(f"  - Labels: {created['labels']}")
    print(f"  - Classes: {classes}")
    return True


def check_kaggle_credentials():
    """Check if Kaggle API credentials are available."""
    kaggle_dir = Path.home() / ".kaggle"
    creds_exist = (kaggle_dir / "kaggle.json").exists()
    if not creds_exist:
        # Also check env vars
        import os

        creds_exist = bool(
            os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        )
    return creds_exist


def check_roboflow_key():
    """Check if Roboflow API key is available."""
    import os

    return bool(os.environ.get("ROBOFLOW_API_KEY"))


def download_kaggle_dataset():
    """Download Dental Cavity Detection Dataset from Kaggle."""
    print("[INFO] Attempting to download Kaggle Dental Cavity Detection Dataset...")
    try:
        import kaggle

        # Dataset: mohamedhanyyy/dental-cavity-detection
        output_dir = RAW_DIR / "kaggle_cavity"
        output_dir.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(
            "mohamedhanyyy/dental-cavity-detection", path=str(output_dir), unzip=True
        )
        print(f"[INFO] Downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        return False


def download_roboflow_dataset():
    """Download dental dataset from Roboflow."""
    print("[INFO] Attempting to download Roboflow dental dataset...")
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
        # Try common dental caries projects
        project = rf.workspace("dental-caries").project("dental-caries-detection")
        dataset = project.version(1).download(
            "yolov8", location=str(RAW_DIR / "roboflow")
        )
        print(f"[INFO] Downloaded Roboflow dataset to {RAW_DIR / 'roboflow'}")
        return True
    except Exception as e:
        print(f"[WARN] Roboflow download failed: {e}")
        return False


def validate_image_count(directory):
    """Count valid images in a directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    count = 0
    broken = []
    if not os.path.exists(directory):
        return 0, broken
    for f in Path(directory).rglob("*"):
        if f.suffix.lower() in valid_exts:
            try:
                img = cv2.imread(str(f))
                if img is not None and img.size > 0:
                    count += 1
                else:
                    broken.append(str(f))
            except Exception:
                broken.append(str(f))
    return count, broken


def validate_annotations(directory):
    """Count annotation files."""
    count = 0
    if not os.path.exists(directory):
        return 0
    for ext in [".txt", ".json", ".xml", ".csv"]:
        for f in Path(directory).rglob(f"*{ext}"):
            count += 1
    return count


def main(skip_dummy=False):
    """Main download workflow."""
    print("=" * 60)
    print("DentalVision-QA Dataset Downloader")
    print("=" * 60)

    # Create directories
    for d in [RAW_DIR, PROCESSED_DIR, ANNOTATIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Check for existing datasets
    raw_count, broken = validate_image_count(RAW_DIR)
    ann_count = validate_annotations(ANNOTATIONS_DIR)

    print(f"\n[INFO] Existing raw images: {raw_count}")
    print(f"[INFO] Existing annotations: {ann_count}")

    if broken:
        print(f"[WARN] Found {len(broken)} broken/corrupted images")

    # Try to download datasets
    downloaded = False

    # Check Kaggle
    if check_kaggle_credentials():
        if download_kaggle_dataset():
            downloaded = True

    # Check Roboflow
    if check_roboflow_key():
        if download_roboflow_dataset():
            downloaded = True

    # Check for Zenodo (manual placement)
    zenodo_path = RAW_DIR / "zenodo_dental"
    if zenodo_path.exists():
        z_count, _ = validate_image_count(zenodo_path)
        if z_count > 0:
            print(f"[INFO] Found Zenodo dataset: {z_count} images")
            downloaded = True

    # Manual dataset check
    manual_path = RAW_DIR / "manual_dataset"
    if manual_path.exists():
        m_count, _ = validate_image_count(manual_path)
        if m_count > 0:
            print(f"[INFO] Found manual dataset: {m_count} images")
            downloaded = True

    if not downloaded and not skip_dummy:
        print("\n[WARN] No dataset found. Creating dummy placeholder dataset...")
        create_dummy_dataset(num_images=20)
    elif downloaded:
        print("\n[SUCCESS] Dataset(s) downloaded successfully!")
    else:
        print("\n[INFO] Skipping dummy dataset creation (--skip-dummy flag present)")

    # Final validation
    print("\n[INFO] Dataset download complete.")
    print("[INFO] Next steps:")
    print("  1. Place Zenodo dataset manually in: data/raw/zenodo_dental/")
    print("  2. Or ensure Kaggle API is configured: kaggle datasets download ...")
    print("  3. Or set ROBOFLOW_API_KEY env var for Roboflow")
    print("  4. Run: python src/prepare_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dental image datasets")
    parser.add_argument(
        "--skip-dummy",
        action="store_true",
        help="Skip dummy dataset creation if no dataset found",
    )
    args = parser.parse_args()
    main(skip_dummy=args.skip_dummy)
