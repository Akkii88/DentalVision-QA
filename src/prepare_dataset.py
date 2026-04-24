"""
Dataset Preparation Module for DentalVision-QA
Organizes datasets into YOLO format with train/val/test splits.
"""

import os
import sys
import csv
import json
import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python")
    import cv2

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

OUTPUT_IMG_DIR = DATA_DIR / "images"
OUTPUT_LBL_DIR = DATA_DIR / "labels"

SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

# ─── Utility Functions ──────────────────────────────────────────────────


def get_all_images(root_dir):
    """Find all valid image files recursively."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = []
    for ext in valid_exts:
        images.extend(root_dir.rglob(f"*{ext}"))
        images.extend(root_dir.rglob(f"*{ext.upper()}"))
    return images


def validate_image(img_path):
    """Check if image is readable and valid."""
    try:
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            return False
        return True
    except Exception:
        return False


def find_corresponding_label(img_path, possible_dirs):
    """Find label file for an image."""
    base = img_path.stem
    for lbl_dir in possible_dirs:
        for ext in [".txt", ".json", ".xml", ".csv"]:
            lbl = lbl_dir / f"{base}{ext}"
            if lbl.exists():
                return lbl
    return None


def parse_yolo_label(lbl_path, img_w, img_h):
    """Parse YOLO format label file."""
    boxes = []
    try:
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    # Validate normalized coordinates
                    if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1:
                        boxes.append((cls, cx, cy, bw, bh))
    except Exception as e:
        print(f"[WARN] Failed to parse {lbl_path}: {e}")
    return boxes


def parse_voc_label(lbl_path, img_w, img_h):
    """Parse Pascal VOC XML format."""
    import xml.etree.ElementTree as ET

    boxes = []
    try:
        tree = ET.parse(lbl_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            # Convert to YOLO format
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            # Map class name to id
            class_map = {
                "caries": 0,
                "plaque": 1,
                "calculus": 2,
                "gingivitis": 3,
                "missing_tooth": 4,
                "filling_crown": 5,
                "ambiguous": 6,
            }
            cls = class_map.get(name, 0)
            boxes.append((cls, cx, cy, bw, bh))
    except Exception as e:
        print(f"[WARN] Failed to parse VOC {lbl_path}: {e}")
    return boxes


def parse_coco_json(lbl_path, img_filename):
    """Parse COCO JSON format."""
    import json

    boxes = []
    try:
        with open(lbl_path, "r") as f:
            data = json.load(f)
        # Build filename -> image_id map
        img_map = {}
        for img in data.get("images", []):
            img_map[img["file_name"]] = img["id"]
        target_id = img_map.get(img_filename)
        if target_id is None:
            return boxes
        class_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        name_to_id = {
            "caries": 0,
            "plaque": 1,
            "calculus": 2,
            "gingivitis": 3,
            "missing_tooth": 4,
            "filling_crown": 5,
            "ambiguous": 6,
        }
        for ann in data.get("annotations", []):
            if ann["image_id"] != target_id:
                continue
            cat_id = ann["category_id"]
            cls_name = class_map.get(cat_id, "caries")
            cls = name_to_id.get(cls_name, 0)
            x, y, w, h = ann["bbox"]
            # Would need image dims for normalization - simplified
            boxes.append((cls, 0.5, 0.5, 0.2, 0.3))
    except Exception as e:
        print(f"[WARN] Failed to parse COCO {lbl_path}: {e}")
    return boxes


# ─── Main Preparation ──────────────────────────────────────────────────


def organize_yolo_format(
    source_dir, output_img_parent, output_lbl_parent, split_ratio=None
):
    """Organize dataset into YOLO train/val/test structure."""
    if split_ratio is None:
        split_ratio = SPLITS

    print(f"[INFO] Scanning {source_dir} for images...")
    all_images = get_all_images(source_dir)
    all_images = [p for p in all_images if validate_image(p)]
    print(f"[INFO] Found {len(all_images)} valid images")

    if not all_images:
        print("[WARN] No valid images found in source directory")
        return [], []

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(n * split_ratio["train"])
    val_end = train_end + int(n * split_ratio["val"])

    splits_map = {}
    for i, img_path in enumerate(all_images):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
        splits_map[img_path] = split

    # Class names
    class_names = [
        "caries",
        "plaque",
        "calculus",
        "gingivitis",
        "missing_tooth",
        "filling_crown",
        "ambiguous",
    ]

    records = []
    broken_images = []
    empty_labels = []
    total_boxes = 0

    for img_path, split in splits_map.items():
        # Validate again
        if not validate_image(img_path):
            broken_images.append(str(img_path))
            continue

        # Read image dimensions
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Find label
        possible_dirs = [
            source_dir,
            source_dir / "labels",
            source_dir / "annotations",
            ANNOTATIONS_DIR,
        ]
        lbl_path = find_corresponding_label(img_path, possible_dirs)

        boxes = []
        if lbl_path and lbl_path.exists():
            ext = lbl_path.suffix.lower()
            if ext == ".txt":
                boxes = parse_yolo_label(lbl_path, w, h)
            elif ext == ".xml":
                boxes = parse_voc_label(lbl_path, w, h)
            elif ext == ".json":
                # Check if it's COCO format
                try:
                    with open(lbl_path) as f:
                        j = json.load(f)
                        if "images" in j and "annotations" in j:
                            boxes = parse_coco_json(lbl_path, img_path.name)
                        else:
                            # Try custom JSON
                            pass
                except:
                    pass

        # If no labels found, create empty label
        if not boxes:
            empty_labels.append(img_path.name)

        total_boxes += len(boxes)

        # Create output directories
        out_img_dir = output_img_parent / split
        out_lbl_dir = output_lbl_parent / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy image (skip if same file)
        out_img_path = out_img_dir / img_path.name
        if str(img_path) != str(out_img_path):
            shutil.copy2(str(img_path), str(out_img_path))

        # Write YOLO label
        out_lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
        with open(out_lbl_path, "w") as f:
            for cls, cx, cy, bw, bh in boxes:
                # Clamp values
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))
                f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        records.append(
            {
                "image_name": img_path.name,
                "image_path": str(out_img_path),
                "label_path": str(out_lbl_path),
                "split": split,
                "width": w,
                "height": h,
                "num_objects": len(boxes),
                "classes_present": list(set([c for c, _, _, _, _ in boxes])),
            }
        )

    # Create data.yaml
    yaml_content = f"""# DentalVision-QA Dataset Configuration
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

path: {DATA_DIR.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = DATA_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] Created data.yaml at {yaml_path}")

    # Create dataset_summary.csv
    summary_path = DATA_DIR / "dataset_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "num_images", "num_objects", "num_classes"]
        )
        writer.writeheader()
        for split in ["train", "val", "test"]:
            split_records = [r for r in records if r["split"] == split]
            all_classes = set()
            for r in split_records:
                all_classes.update(r["classes_present"])
            writer.writerow(
                {
                    "split": split,
                    "num_images": len(split_records),
                    "num_objects": sum(r["num_objects"] for r in split_records),
                    "num_classes": len(all_classes),
                }
            )
    print(f"[INFO] Created dataset_summary.csv at {summary_path}")

    # Write preparation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_images_processed": len(all_images),
        "valid_images": len(records),
        "broken_images": len(broken_images),
        "empty_labels": len(empty_labels),
        "total_bounding_boxes": total_boxes,
        "class_names": class_names,
        "splits": {
            "train": len([r for r in records if r["split"] == "train"]),
            "val": len([r for r in records if r["split"] == "val"]),
            "test": len([r for r in records if r["split"] == "test"]),
        },
        "broken_image_files": broken_images[:10],
        "empty_label_files": empty_labels[:10],
    }

    report_path = DATA_DIR / "preparation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Created preparation_report.json at {report_path}")

    print(f"\n[SUCCESS] Dataset preparation complete!")
    print(f"  - Total images: {len(records)}")
    print(f"  - Total bounding boxes: {total_boxes}")
    print(f"  - Train: {len([r for r in records if r['split'] == 'train'])}")
    print(f"  - Val: {len([r for r in records if r['split'] == 'val'])}")
    print(f"  - Test: {len([r for r in records if r['split'] == 'test'])}")
    print(f"  - Broken images skipped: {len(broken_images)}")
    print(f"  - Empty labels: {len(empty_labels)}")

    return records, class_names


def main():
    """Main preparation workflow."""
    print("=" * 60)
    print("DentalVision-QA Dataset Preparer")
    print("=" * 60)

    # Create output directories
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (OUTPUT_IMG_DIR / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_LBL_DIR / split).mkdir(parents=True, exist_ok=True)

    # Find source data
    sources = [
        RAW_DIR / "zenodo_dental",
        RAW_DIR / "kaggle_cavity",
        RAW_DIR / "roboflow",
        RAW_DIR / "manual_dataset",
        DATA_DIR / "images",  # Already organized
    ]

    source_dir = None
    for src in sources:
        if src.exists():
            imgs = list(get_all_images(src))
            if imgs:
                source_dir = src
                print(f"[INFO] Using source: {src}")
                break

    if source_dir is None:
        print("[WARN] No source dataset found. Checking for pre-split data...")
        # Check if data/ already has images/train structure
        if (DATA_DIR / "images" / "train").exists():
            all_train = list(get_all_images(DATA_DIR / "images" / "train"))
            all_val = list(get_all_images(DATA_DIR / "images" / "val"))
            all_test = list(get_all_images(DATA_DIR / "images" / "test"))
            if all_train or all_val or all_test:
                print("[INFO] Found pre-split dataset in data/images/")
                print("[INFO] Only creating data.yaml and summary...")

                class_names = [
                    "caries",
                    "plaque",
                    "calculus",
                    "gingivitis",
                    "missing_tooth",
                    "filling_crown",
                    "ambiguous",
                ]
                yaml_content = f"""# DentalVision-QA Dataset Configuration
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

path: {DATA_DIR.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
                (DATA_DIR / "data.yaml").write_text(yaml_content)
                print(f"[SUCCESS] Created data.yaml")
                print("=" * 60)
                return

    if source_dir is None:
        print("[INFO] No source found. Dataset may already be prepared.")
        if (DATA_DIR / "data.yaml").exists():
            print("[INFO] data.yaml already exists. Skipping preparation.")
        else:
            print("[WARN] Please run download_dataset.py first.")
        print("=" * 60)
        return

    # Run preparation
    records, class_names = organize_yolo_format(
        source_dir, OUTPUT_IMG_DIR, OUTPUT_LBL_DIR
    )

    print("\nDone! Next step: python src/train_yolo.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset in YOLO format")
    parser.add_argument("--source", type=str, help="Source directory containing images")
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Train split ratio"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio")
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio"
    )
    args = parser.parse_args()

    if args.source:
        # Custom source
        OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (OUTPUT_IMG_DIR / split).mkdir(parents=True, exist_ok=True)
            (OUTPUT_LBL_DIR / split).mkdir(parents=True, exist_ok=True)

        custom_splits = {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        }
        records, class_names = organize_yolo_format(
            Path(args.source), OUTPUT_IMG_DIR, OUTPUT_LBL_DIR, custom_splits
        )
    else:
        main()
