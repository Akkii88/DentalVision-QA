"""
Prediction Module for DentalVision-QA
Runs inference and saves prediction results.
"""

import os
import sys
import json
import argparse
import csv
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

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    os.system("pip install pandas")
    import pandas as pd

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

CLASS_NAMES = [
    "caries",
    "plaque",
    "calculus",
    "gingivitis",
    "missing_tooth",
    "filling_crown",
    "ambiguous",
]

COLOR_PALETTE = [
    (255, 0, 0),  # caries - red
    (0, 255, 0),  # plaque - green
    (0, 0, 255),  # calculus - blue
    (255, 255, 0),  # gingivitis - cyan
    (255, 0, 255),  # missing_tooth - magenta
    (0, 255, 255),  # filling_crown - yellow
    (128, 128, 128),  # ambiguous - gray
]


def draw_detections(img, boxes, class_names, colors, conf_threshold=0.3):
    """Draw bounding boxes and labels on image."""
    img_vis = img.copy()
    h, w = img.shape[:2]

    for box in boxes:
        cls_id = int(box[0])
        conf = box[1]
        cx, cy, bw, bh = box[2:6]

        if conf < conf_threshold:
            continue

        # Convert to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = colors[cls_id % len(colors)]
        label = f"{class_names[cls_id] if cls_id < len(class_names) else 'unknown'}: {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_vis, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(
            img_vis,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return img_vis


def run_predictions(
    input_dir=None, model_path=None, output_dir=None, conf_threshold=0.3
):
    """Run predictions on images and save results."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = PREDICTIONS_DIR / "annotated_images"
        output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    if model_path is None:
        model_dirs = list(MODEL_DIR.glob("dental_detector"))
        if not model_dirs:
            print("[WARN] No trained model found. Using yolov8n.pt")
            model = YOLO("yolov8n.pt")
        else:
            best_model = model_dirs[0] / "weights" / "best.pt"
            if best_model.exists():
                model = YOLO(str(best_model))
                print(f"[INFO] Loaded trained model: {best_model}")
            else:
                last_model = model_dirs[0] / "weights" / "last.pt"
                if last_model.exists():
                    model = YOLO(str(last_model))
                    print(f"[INFO] Loaded model: {last_model}")
                else:
                    model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)

    # Determine input directory
    if input_dir is None:
        input_path = DATA_DIR / "images" / "test"
        if not input_path.exists() or not any(input_path.iterdir()):
            input_path = DATA_DIR / "images" / "val"
        if not input_path.exists() or not any(input_path.iterdir()):
            input_path = DATA_DIR / "images" / "train"
    else:
        input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Get all images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = []
    for ext in image_exts:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    print("=" * 60)
    print("DentalVision-QA Prediction Module")
    print("=" * 60)
    print(f"Model: {model_path or 'trained model'}")
    print(f"Input: {input_path}")
    print(f"Images: {len(image_files)}")
    print(f"Confidence threshold: {conf_threshold}")
    print("=" * 60)

    # Run predictions
    predictions = []

    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        # Run inference
        results = model(str(img_path), verbose=False, conf=conf_threshold)

        if results and len(results) > 0:
            r = results[0]
            boxes_data = []

            if hasattr(r, "boxes") and r.boxes is not None:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else 0
                    conf = (
                        float(box.conf.cpu().numpy()[0])
                        if hasattr(box, "conf")
                        else 0.0
                    )
                    xyxy = (
                        box.xyxy.cpu().numpy()[0]
                        if hasattr(box, "xyxy")
                        else [0, 0, 0, 0]
                    )

                    h_img, w_img = img.shape[:2]
                    x1, y1, x2, y2 = xyxy

                    # Convert to YOLO format (normalized)
                    cx = ((x1 + x2) / 2) / w_img
                    cy = ((y1 + y2) / 2) / h_img
                    bw = (x2 - x1) / w_img
                    bh = (y2 - y1) / h_img

                    boxes_data.append(
                        {
                            "cls_id": cls_id,
                            "class": CLASS_NAMES[cls_id]
                            if cls_id < len(CLASS_NAMES)
                            else "unknown",
                            "confidence": conf,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "cx": cx,
                            "cy": cy,
                            "bw": bw,
                            "bh": bh,
                        }
                    )

                    predictions.append(
                        {
                            "image_name": img_path.name,
                            "predicted_class": CLASS_NAMES[cls_id]
                            if cls_id < len(CLASS_NAMES)
                            else "unknown",
                            "confidence": round(conf, 4),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                            "status": "detected",
                        }
                    )

            # Draw on image if boxes found
            if boxes_data:
                img_vis = draw_detections(
                    img,
                    [
                        (
                            b["cls_id"],
                            b["confidence"],
                            b["cx"],
                            b["cy"],
                            b["bw"],
                            b["bh"],
                        )
                        for b in boxes_data
                    ],
                    CLASS_NAMES,
                    COLOR_PALETTE,
                    conf_threshold,
                )

                # Save annotated image
                out_path = output_path / f"pred_{img_path.name}"
                cv2.imwrite(str(out_path), img_vis)
            else:
                # No detections - save original with "no findings" label
                img_vis = img.copy()
                cv2.putText(
                    img_vis,
                    "No findings",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                out_path = output_path / f"pred_{img_path.name}"
                cv2.imwrite(str(out_path), img_vis)

                predictions.append(
                    {
                        "image_name": img_path.name,
                        "predicted_class": "none",
                        "confidence": 0.0,
                        "x1": 0,
                        "y1": 0,
                        "x2": 0,
                        "y2": 0,
                        "status": "no_detection",
                    }
                )
        else:
            # No results
            img_vis = img.copy()
            cv2.putText(
                img_vis,
                "No findings",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            out_path = output_path / f"pred_{img_path.name}"
            cv2.imwrite(str(out_path), img_vis)

            predictions.append(
                {
                    "image_name": img_path.name,
                    "predicted_class": "none",
                    "confidence": 0.0,
                    "x1": 0,
                    "y1": 0,
                    "x2": 0,
                    "y2": 0,
                    "status": "no_detection",
                }
            )

    # Save predictions CSV
    csv_path = PREDICTIONS_DIR / "predictions.csv"
    df = pd.DataFrame(predictions)
    df.to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] Predictions complete!")
    print(f"  - Processed: {len(predictions)} images")
    print(f"  - Annotated images: {output_path}")
    print(f"  - Predictions CSV: {csv_path}")
    print(f"  - Avg confidence: {df['confidence'].mean():.4f}")
    print(f"  - Class distribution:")
    for cls in df["predicted_class"].unique():
        cnt = len(df[df["predicted_class"] == cls])
        print(f"    - {cls}: {cnt}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run predictions on dental images")
    parser.add_argument(
        "--input", type=str, default=None, help="Input directory with images"
    )
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for annotated images"
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    try:
        predictions = run_predictions(
            input_dir=args.input,
            model_path=args.model,
            output_dir=args.output,
            conf_threshold=args.conf,
        )
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
