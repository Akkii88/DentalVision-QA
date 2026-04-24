"""
Utility Module for DentalVision-QA
Shared helper functions and utilities.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np


# ─── Path Utilities ─────────────────────────────────────────────────────


def get_project_root(current_file=__file__):
    """Get the project root directory."""
    return Path(current_file).parent.parent


def ensure_dir(path):
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# ─── File Utilities ────────────────────────────────────────────────────


def calculate_file_hash(filepath, algorithm="md5"):
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def copy_with_backup(src, dst):
    """Copy file, backup destination if exists."""
    if os.path.exists(dst):
        backup = f"{dst}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(dst, backup)
        print(f"[INFO] Backed up {dst} to {backup}")
    shutil.copy2(src, dst)


def safe_remove(filepath):
    """Safely remove a file."""
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        print(f"[WARN] Could not remove {filepath}: {e}")
        return False


# ─── Data Utilities ────────────────────────────────────────────────────


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data, filepath, indent=2):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)


def load_yaml(filepath):
    """Load YAML file."""
    try:
        import yaml

        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Installing pyyaml...")
        os.system("pip install pyyaml")
        import yaml

        with open(filepath, "r") as f:
            return yaml.safe_load(f)


def save_yaml(data, filepath):
    """Save data to YAML file."""
    try:
        import yaml

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    except ImportError:
        print("Installing pyyaml...")
        os.system("pip install pyyaml")
        import yaml

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# ─── YOLO Utilities ────────────────────────────────────────────────────


def yolo_to_coco(yolo_box, img_width, img_height):
    """Convert YOLO format to COCO format."""
    cls, cx, cy, bw, bh = yolo_box
    x = (cx - bw / 2) * img_width
    y = (cy - bh / 2) * img_height
    w = bw * img_width
    h = bh * img_height
    return cls, x, y, w, h


def coco_to_yolo(coco_box, img_width, img_height):
    """Convert COCO format to YOLO format."""
    x, y, w, h = coco_box[:4]
    cls = coco_box[0] if len(coco_box) > 4 else 0
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    bw = w / img_width
    bh = h / img_height
    return cls, cx, cy, bw, bh


def compute_iou(box1, box2, format="yolo"):
    """Compute IoU between two boxes."""
    if format == "yolo":
        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2
        x1_min = cx1 - w1 / 2
        y1_min = cy1 - h1 / 2
        x1_max = cx1 + w1 / 2
        y1_max = cy1 + h1 / 2
        x2_min = cx2 - w2 / 2
        y2_min = cy2 - h2 / 2
        x2_max = cx2 + w2 / 2
        y2_max = cy2 + h2 / 2
    else:
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ─── Color Utilities ───────────────────────────────────────────────────


def get_class_colors(num_classes=7):
    """Generate distinct colors for classes."""
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 128, 128),  # Gray
    ]
    return colors[:num_classes]


def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB to hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ─── Progress Utilities ────────────────────────────────────────────────


class ProgressTracker:
    """Track and display progress of operations."""

    def __init__(self, total, description="Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, increment=1):
        """Update progress."""
        self.current += increment
        percent = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        print(
            f"\r{self.description}: {self.current}/{self.total} "
            f"({percent:.1f}%) ETA: {eta:.0f}s",
            end="",
            flush=True,
        )
        if self.current >= self.total:
            print()  # New line

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.current < self.total:
            print()


# ─── Main Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DentalVision-QA Utilities")
    print("=" * 40)

    # Test color utilities
    colors = get_class_colors()
    print(f"Class colors: {colors}")

    # Test IoU
    box1 = (0.5, 0.5, 0.2, 0.3)
    box2 = (0.5, 0.5, 0.2, 0.3)
    iou = compute_iou(box1, box2)
    print(f"IoU test: {iou}")

    print("\nAll utility functions loaded successfully!")
