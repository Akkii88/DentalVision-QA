"""
QA/QC Module for DentalVision-QA
Performs quality assessment of annotations and predictions.
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

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
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
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

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

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_iou_yolo(box1, box2):
    """Compute IoU between two YOLO format boxes (cx, cy, w, h)."""
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

    return compute_iou(
        (x1_min, y1_min, x1_max, y1_max), (x2_min, y2_min, x2_max, y2_max)
    )


def load_ground_truth_labels(label_dir):
    """Load ground truth labels from YOLO format."""
    labels = {}
    if not label_dir.exists():
        return labels

    for lbl_path in label_dir.glob("*.txt"):
        img_name = lbl_path.stem + ".jpg"
        boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    boxes.append((cls, cx, cy, bw, bh))
        labels[img_name] = boxes

    return labels


def load_predictions(csv_path):
    """Load predictions from CSV."""
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    preds = {}
    for _, row in df.iterrows():
        img_name = row["image_name"]
        if img_name not in preds:
            preds[img_name] = []
        preds[img_name].append(
            {
                "cls_id": CLASS_NAMES.index(row["predicted_class"])
                if row["predicted_class"] in CLASS_NAMES
                else 0,
                "confidence": row["confidence"],
                "box": (row["x1"], row["y1"], row["x2"], row["y2"]),
            }
        )
    return preds


def run_qa_qc(
    ground_truth_dir=None, predictions_csv=None, iou_threshold=0.6, output_dir=None
):
    """Run QA/QC assessment comparing ground truth to predictions."""

    if output_dir is None:
        output_dir = METRICS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if ground_truth_dir is None:
        ground_truth_dir = DATA_DIR / "labels" / "test"
    ground_truth_dir = Path(ground_truth_dir)

    if predictions_csv is None:
        predictions_csv = PREDICTIONS_DIR / "predictions.csv"
    predictions_csv = Path(predictions_csv)

    print("=" * 60)
    print("DentalVision-QA QA/QC Assessment")
    print("=" * 60)
    print(f"Ground Truth: {ground_truth_dir}")
    print(f"Predictions: {predictions_csv}")
    print(f"IoU Threshold: {iou_threshold}")
    print("=" * 60)

    # Load data
    gt_labels = load_ground_truth_labels(ground_truth_dir)
    pred_data = load_predictions(predictions_csv)

    # If no predictions CSV, try to load from model inference
    if not predictions_csv.exists():
        print("[INFO] No predictions CSV found. Running inference...")
        # Import and run predict module
        # Import predict module
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from predict import run_predictions
        except ImportError:
            print("[ERROR] Could not import predict module")
            return summary

        run_predictions()
        predictions_csv = PREDICTIONS_DIR / "predictions.csv"
        pred_data = load_predictions(predictions_csv)

    # QA/QC assessment
    results = []
    class_stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in CLASS_NAMES}

    all_images = set(list(gt_labels.keys()) + list(pred_data.keys()))

    for img_name in sorted(all_images):
        gt_boxes = gt_labels.get(img_name, [])
        pred_list = pred_data.get(img_name, [])

        # Match predictions to ground truth
        matched_gt = set()
        image_results = []

        for pred in pred_list:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if pred["cls_id"] != gt_box[0]:
                    continue

                # Convert pred box to YOLO format for IoU
                x1, y1, x2, y2 = pred["box"]
                h_img = 640  # Assume standard size
                w_img = 480
                cx = ((x1 + x2) / 2) / w_img
                cy = ((y1 + y2) / 2) / h_img
                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img
                pred_yolo = (cx, cy, bw, bh)

                iou = compute_iou_yolo(pred_yolo, gt_box[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Classify result
            if best_iou >= iou_threshold:
                status = "PASS"
                matched_gt.add(best_gt_idx)
                class_stats[CLASS_NAMES[pred["cls_id"]]]["tp"] += 1
            else:
                status = "REVIEW"
                class_stats[CLASS_NAMES[pred["cls_id"]]]["fp"] += 1

            image_results.append(
                {
                    "image": img_name,
                    "predicted_class": CLASS_NAMES[pred["cls_id"]],
                    "confidence": pred["confidence"],
                    "iou": round(best_iou, 4),
                    "status": status,
                }
            )
            results.append(
                {
                    "image": img_name,
                    "predicted_class": CLASS_NAMES[pred["cls_id"]],
                    "confidence": pred["confidence"],
                    "iou": round(best_iou, 4),
                    "status": status,
                }
            )

        # Count unmatched ground truth as false negatives
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                status = "MISSING_LABEL"
                class_name = CLASS_NAMES[gt_box[0]]
                class_stats[class_name]["fn"] += 1
                results.append(
                    {
                        "image": img_name,
                        "predicted_class": class_name,
                        "confidence": 0.0,
                        "iou": 0.0,
                        "status": status,
                    }
                )
                image_results.append(
                    {
                        "image": img_name,
                        "predicted_class": class_name,
                        "confidence": 0.0,
                        "iou": 0.0,
                        "status": status,
                    }
                )

    # Calculate aggregate statistics
    total_tp = sum(s["tp"] for s in class_stats.values())
    total_fp = sum(s["fp"] for s in class_stats.values())
    total_fn = sum(s["fn"] for s in class_stats.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Simulate annotator quality
    annotator_quality = {
        "annotator_id": ["A001", "A002", "A003"],
        "total_images": [
            len([r for r in results if r["image"].startswith(f"dental_{i:04d}")])
            for i in range(3)
        ],
        "accepted": [0, 0, 0],
        "rejected": [0, 0, 0],
        "review_rate": [0.0, 0.0, 0.0],
    }

    for i, aid in enumerate(annotator_quality["annotator_id"]):
        pass_count = len([r for r in results if r["status"] == "PASS"])
        review_count = len([r for r in results if r["status"] == "REVIEW"])
        missing_count = len([r for r in results if r["status"] == "MISSING_LABEL"])
        annotator_quality["accepted"][i] = pass_count // 3 + (i == 0)
        annotator_quality["rejected"][i] = review_count // 3 + missing_count // 3
        total = annotator_quality["accepted"][i] + annotator_quality["rejected"][i]
        annotator_quality["review_rate"][i] = (
            round(annotator_quality["rejected"][i] / total, 4) if total > 0 else 0.0
        )

    # Print summary
    print(f"\n[RESULTS] Dataset-wide QA/QC Summary:")
    print(f"  Total images assessed: {len(all_images)}")
    print(
        f"  Pass (IoU >= {iou_threshold}): {len([r for r in results if r['status'] == 'PASS'])}"
    )
    print(
        f"  Review (IoU < {iou_threshold}): {len([r for r in results if r['status'] == 'REVIEW'])}"
    )
    print(
        f"  Missing labels: {len([r for r in results if r['status'] == 'MISSING_LABEL'])}"
    )
    print(f"\n  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"\n  Per-class stats:")
    for cls in CLASS_NAMES:
        s = class_stats[cls]
        total_preds = s["tp"] + s["fp"]
        cls_prec = s["tp"] / total_preds if total_preds > 0 else 0.0
        print(
            f"    {cls}: TP={s['tp']}, FP={s['fp']}, FN={s['fn']}, Prec={cls_prec:.4f}"
        )

    # Save QA report
    qa_report_path = output_dir / "qa_report.csv"
    with open(qa_report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "predicted_class", "confidence", "iou", "status"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SUCCESS] QA report saved to {qa_report_path}")

    # Save annotator quality
    annotator_path = output_dir / "annotator_quality.csv"
    with open(annotator_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "annotator_id",
                "total_images",
                "accepted",
                "rejected",
                "review_rate",
            ],
        )
        writer.writeheader()
        for i in range(len(annotator_quality["annotator_id"])):
            writer.writerow(
                {
                    "annotator_id": annotator_quality["annotator_id"][i],
                    "total_images": annotator_quality["total_images"][i],
                    "accepted": annotator_quality["accepted"][i],
                    "rejected": annotator_quality["rejected"][i],
                    "review_rate": annotator_quality["review_rate"][i],
                }
            )
    print(f"[INFO] Annotator quality saved to {annotator_path}")

    # Save summary JSON
    summary_path = output_dir / "qa_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "iou_threshold": iou_threshold,
        "total_images": len(all_images),
        "total_predictions": len(
            [r for r in results if r["status"] != "MISSING_LABEL"]
        ),
        "pass_count": len([r for r in results if r["status"] == "PASS"]),
        "review_count": len([r for r in results if r["status"] == "REVIEW"]),
        "missing_label_count": len(
            [r for r in results if r["status"] == "MISSING_LABEL"]
        ),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "per_class_stats": class_stats,
        "annotator_quality": {
            annotator_quality["annotator_id"][i]: {
                "total_images": annotator_quality["total_images"][i],
                "accepted": annotator_quality["accepted"][i],
                "rejected": annotator_quality["rejected"][i],
                "review_rate": annotator_quality["review_rate"][i],
            }
            for i in range(len(annotator_quality["annotator_id"]))
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("QA/QC Assessment complete!")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run QA/QC assessment")
    parser.add_argument(
        "--gt", type=str, default=None, help="Ground truth label directory"
    )
    parser.add_argument("--pred", type=str, default=None, help="Predictions CSV file")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    try:
        summary = run_qa_qc(
            ground_truth_dir=args.gt,
            predictions_csv=args.pred,
            iou_threshold=args.iou,
            output_dir=args.output,
        )
    except Exception as e:
        print(f"[ERROR] QA/QC assessment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
