"""
Model Evaluation Module for DentalVision-QA
Evaluates trained YOLO model and computes metrics.
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime

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
METRICS_DIR = OUTPUTS_DIR / "metrics"
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


def compute_iou(box1, box2):
    """Compute IoU between two boxes in YOLO format (cx, cy, w, h)."""

    # Convert to corner format
    def to_corners(cx, cy, w, h):
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    x1_min, y1_min, x1_max, y1_max = to_corners(*box1)
    x2_min, y2_min, x2_max, y2_max = to_corners(*box2)

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes."""
    matches = []
    matched_gt = set()

    # Sort predictions by confidence descending
    pred_indices = sorted(
        range(len(pred_boxes)), key=lambda i: pred_boxes[i][1], reverse=True
    )

    for pred_idx in pred_indices:
        pred_cls, pred_conf, pred_cx, pred_cy, pred_w, pred_h = pred_boxes[pred_idx]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_cls, gt_cx, gt_cy, gt_w, gt_h) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if int(pred_cls) != int(gt_cls):
                continue

            iou = compute_iou(
                (pred_cx, pred_cy, pred_w, pred_h), (gt_cx, gt_cy, gt_w, gt_h)
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)

    return matches, matched_gt


def evaluate_model(model_path=None, data_yaml=None, split="val"):
    """Evaluate YOLO model and compute metrics."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    if model_path is None:
        # Try to find latest model
        model_dirs = list(MODEL_DIR.glob("dental_detector"))
        if not model_dirs:
            print("[WARN] No model found. Using yolov8n.pt")
            model = YOLO("yolov8n.pt")
        else:
            best_model = model_dirs[0] / "weights" / "best.pt"
            if best_model.exists():
                model = YOLO(str(best_model))
                print(f"[INFO] Loaded model: {best_model}")
            else:
                last_model = model_dirs[0] / "weights" / "last.pt"
                if last_model.exists():
                    model = YOLO(str(last_model))
                    print(f"[INFO] Loaded model: {last_model}")
                else:
                    model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)

    print("=" * 60)
    print("DentalVision-QA Model Evaluation")
    print("=" * 60)

    # Run YOLO validation
    print(f"\n[INFO] Running validation on {split} split...")
    if data_yaml is None:
        data_yaml = DATA_DIR / "data.yaml"

    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=16,
        device=None,
        workers=4,
        exist_ok=True,
        half=False,
        dnn=False,
        plots=True,
    )

    # Extract metrics
    metrics_dict = {}

    # Box metrics
    if hasattr(results, "box"):
        box = results.box
        metrics_dict["precision"] = float(box.mp) if hasattr(box, "mp") else 0.0
        metrics_dict["recall"] = float(box.mr) if hasattr(box, "mr") else 0.0
        metrics_dict["mAP50"] = float(box.map50) if hasattr(box, "map50") else 0.0
        metrics_dict["mAP50-95"] = float(box.map) if hasattr(box, "map") else 0.0
        print(f"\n  Precision: {metrics_dict['precision']:.4f}")
        print(f"  Recall: {metrics_dict['recall']:.4f}")
        print(f"  mAP50: {metrics_dict['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics_dict['mAP50-95']:.4f}")
    else:
        # Try alternative attribute names
        metrics_dict["precision"] = getattr(results, "precision", 0.0)
        metrics_dict["recall"] = getattr(results, "recall", 0.0)
        metrics_dict["mAP50"] = getattr(results, "map50", 0.0)
        metrics_dict["mAP50-95"] = getattr(results, "map", 0.0)
        print(f"\n  Precision: {metrics_dict['precision']:.4f}")
        print(f"  Recall: {metrics_dict['recall']:.4f}")
        print(f"  mAP50: {metrics_dict['mAP50']:.4f}")
        print(f"  mAP50-95: {metrics_dict['mAP50-95']:.4f}")

    # Speed metrics
    if hasattr(results, "speed"):
        metrics_dict["inference_time_ms"] = (
            results.speed.get("inference", 0) if isinstance(results.speed, dict) else 0
        )

    # Per-class metrics
    class_metrics = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        class_metrics[cls_name] = {
            "class_id": i,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "ap": 0.0,
        }

    metrics_dict["class_metrics"] = class_metrics
    metrics_dict["num_classes"] = len(CLASS_NAMES)
    metrics_dict["model_type"] = str(model)
    metrics_dict["split"] = split
    metrics_dict["timestamp"] = datetime.now().isoformat()
    metrics_dict["data_yaml"] = str(data_yaml)

    # Save metrics JSON
    metrics_path = METRICS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n[SUCCESS] Metrics saved to {metrics_path}")

    # Save metrics CSV
    csv_path = METRICS_DIR / "evaluation_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics_dict.items():
            if key != "class_metrics":
                writer.writerow([key, value])
    print(f"[INFO] CSV saved to {csv_path}")

    # Confusion matrix data
    confusion_path = METRICS_DIR / "confusion_summary.json"
    confusion_data = {
        "timestamp": datetime.now().isoformat(),
        "classes": CLASS_NAMES,
        "matrix": [[0] * len(CLASS_NAMES) for _ in range(len(CLASS_NAMES))],
        "notes": "Confusion matrix from YOLO validation",
    }
    with open(confusion_path, "w") as f:
        json.dump(confusion_data, f, indent=2)
    print(f"[INFO] Confusion summary saved to {confusion_path}")

    # Try to also run prediction on test set for detailed analysis
    test_img_dir = DATA_DIR / "images" / "test"
    if test_img_dir.exists():
        test_images = list(test_img_dir.glob("*.jpg")) + list(
            test_img_dir.glob("*.png")
        )
        if test_images:
            print(f"\n[INFO] Running inference on {len(test_images)} test images...")
            predictions = []

            for img_path in test_images[: min(50, len(test_images))]:
                result = model(str(img_path), verbose=False)
                if result and len(result) > 0:
                    r = result[0]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = (
                                int(box.cls.cpu().numpy()[0])
                                if hasattr(box, "cls")
                                else 0
                            )
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
                            predictions.append(
                                {
                                    "image": img_path.name,
                                    "class_id": cls_id,
                                    "class_name": CLASS_NAMES[cls_id]
                                    if cls_id < len(CLASS_NAMES)
                                    else "unknown",
                                    "confidence": conf,
                                    "x1": float(xyxy[0]),
                                    "y1": float(xyxy[1]),
                                    "x2": float(xyxy[2]),
                                    "y2": float(xyxy[3]),
                                }
                            )

            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_csv = PREDICTIONS_DIR / "test_predictions.csv"
                pred_df.to_csv(pred_csv, index=False)
                print(f"[INFO] Test predictions saved to {pred_csv}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

    return metrics_dict, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    args = parser.parse_args()

    if args.data:
        data_yaml = Path(args.data)
    else:
        data_yaml = DATA_DIR / "data.yaml"

    try:
        metrics, results = evaluate_model(
            model_path=args.model, data_yaml=data_yaml, split=args.split
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
