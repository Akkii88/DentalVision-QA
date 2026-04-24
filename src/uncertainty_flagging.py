"""
Uncertainty Flagging Module for DentalVision-QA
Flags cases requiring clinical review.
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    os.system("pip install pandas")
    import pandas as pd

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"

CLASS_NAMES = [
    "caries",
    "plaque",
    "calculus",
    "gingivitis",
    "missing_tooth",
    "filling_crown",
    "ambiguous",
]


def flag_low_confidence(df, threshold=0.70):
    """Flag predictions with confidence below threshold."""
    low_conf = df[df["confidence"] < threshold].copy()
    low_conf["flag_reason"] = "LOW_CONFIDENCE"
    low_conf["flag_severity"] = "HIGH" if threshold == 0.50 else "MEDIUM"
    return low_conf


def flag_overlapping_predictions(df, iou_threshold=0.5):
    """Flag multiple overlapping predictions for same image."""
    overlapping = []

    for img_name in df["image_name"].unique():
        img_preds = df[df["image_name"] == img_name].copy()
        if len(img_preds) < 2:
            continue

        preds_list = img_preds.to_dict("records")

        for i in range(len(preds_list)):
            for j in range(i + 1, len(preds_list)):
                p1 = preds_list[i]
                p2 = preds_list[j]

                # Compute IoU
                x1_min, y1_min, x1_max, y1_max = p1["x1"], p1["y1"], p1["x2"], p1["y2"]
                x2_min, y2_min, x2_max, y2_max = p2["x1"], p2["y1"], p2["x2"], p2["y2"]

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
                    continue

                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                union_area = box1_area + box2_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0.0

                if iou > iou_threshold:
                    overlapping.append(
                        {
                            "image_name": img_name,
                            "predicted_class": f"{p1['predicted_class']} vs {p2['predicted_class']}",
                            "confidence": max(p1["confidence"], p2["confidence"]),
                            "x1": min(p1["x1"], p2["x1"]),
                            "y1": min(p1["y1"], p2["y1"]),
                            "x2": max(p1["x2"], p2["x2"]),
                            "y2": max(p1["y2"], p2["y2"]),
                            "status": "OVERLAP",
                            "flag_reason": "OVERLAPPING_PREDICTIONS",
                            "flag_severity": "MEDIUM",
                            "iou": round(iou, 4),
                        }
                    )

    return pd.DataFrame(overlapping)


def flag_ambiguous_class(df):
    """Flag predictions of 'ambiguous' class."""
    ambiguous = df[df["predicted_class"] == "ambiguous"].copy()
    ambiguous["flag_reason"] = "AMBIGUOUS_CLASS"
    ambiguous["flag_severity"] = "LOW"
    return ambiguous


def flag_large_bbox(df, size_threshold=0.5):
    """Flag unusually large bounding boxes."""
    # Assuming normalized coordinates 0-1
    df["bbox_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    large = df[df["bbox_area"] > size_threshold].copy()
    large["flag_reason"] = "LARGE_BOUNDING_BOX"
    large["flag_severity"] = "LOW"
    return large


def flag_edge_cases(df, image_width=480, image_height=640):
    """Flag predictions near image edges."""
    margin = 0.05  # 5% margin
    edge = df[
        (df["x1"] < margin * image_width)
        | (df["y1"] < margin * image_height)
        | (df["x2"] > (1 - margin) * image_width)
        | (df["y2"] > (1 - margin) * image_height)
    ].copy()
    edge["flag_reason"] = "EDGE_DETECTION"
    edge["flag_severity"] = "LOW"
    return edge


def run_uncertainty_flagging(
    predictions_csv=None,
    confidence_threshold=0.70,
    iou_overlap_threshold=0.5,
    output_dir=None,
):
    """Main uncertainty flagging workflow."""

    if output_dir is None:
        output_dir = METRICS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if predictions_csv is None:
        predictions_csv = PREDICTIONS_DIR / "predictions.csv"
    predictions_csv = Path(predictions_csv)

    print("=" * 60)
    print("DentalVision-QA Uncertainty Flagging")
    print("=" * 60)
    print(f"Predictions: {predictions_csv}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Overlap IoU threshold: {iou_overlap_threshold}")
    print("=" * 60)

    # Load predictions
    if not predictions_csv.exists():
        print("[WARN] Predictions CSV not found. Running prediction module...")
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from predict import run_predictions

            run_predictions()
        except ImportError:
            print("[ERROR] Could not import predict module")
        predictions_csv = PREDICTIONS_DIR / "predictions.csv"

    df = pd.read_csv(predictions_csv)
    print(f"\n[INFO] Loaded {len(df)} predictions")

    # Apply flagging rules
    print("\n[INFO] Applying flagging rules...")

    # 1. Low confidence
    low_conf_df = flag_low_confidence(df, threshold=confidence_threshold)
    print(f"  - Low confidence (<{confidence_threshold}): {len(low_conf_df)}")

    # 2. Overlapping predictions
    overlap_df = flag_overlapping_predictions(df, iou_threshold=iou_overlap_threshold)
    print(f"  - Overlapping predictions: {len(overlap_df)}")

    # 3. Ambiguous class
    ambig_df = flag_ambiguous_class(df)
    print(f"  - Ambiguous class: {len(ambig_df)}")

    # 4. Large bounding boxes
    large_df = flag_large_bbox(df, size_threshold=0.4)
    print(f"  - Large bounding boxes (>40%): {len(large_df)}")

    # 5. Edge cases
    edge_df = flag_edge_cases(df)
    print(f"  - Edge detections: {len(edge_df)}")

    # Combine all flagged cases
    all_flagged = pd.concat(
        [low_conf_df, overlap_df, ambig_df, large_df, edge_df], ignore_index=True
    )

    # Remove duplicates (same image, same class, similar location)
    if len(all_flagged) > 0:
        all_flagged = all_flagged.drop_duplicates(
            subset=["image_name", "predicted_class", "x1", "y1", "x2", "y2"],
            keep="first",
        )
        all_flagged = all_flagged.sort_values("confidence").reset_index(drop=True)

    print(f"\n[TOTAL] Unique flagged cases: {len(all_flagged)}")

    # Group by severity
    if len(all_flagged) > 0:
        severity_counts = all_flagged["flag_severity"].value_counts().to_dict()
        print(f"\n  By severity:")
        for sev in ["HIGH", "MEDIUM", "LOW"]:
            count = severity_counts.get(sev, 0)
            print(f"    {sev}: {count}")

        print(f"\n  By reason:")
        reason_counts = all_flagged["flag_reason"].value_counts().to_dict()
        for reason, count in reason_counts.items():
            print(f"    {reason}: {count}")

    # Save flagged cases
    flagged_path = output_dir / "flagged_cases.csv"
    all_flagged.to_csv(flagged_path, index=False)
    print(f"\n[SUCCESS] Flagged cases saved to {flagged_path}")

    # Save summary JSON
    summary_path = output_dir / "uncertainty_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(df),
        "total_flagged": len(all_flagged),
        "flag_rate": round(len(all_flagged) / len(df), 4) if len(df) > 0 else 0.0,
        "confidence_threshold": confidence_threshold,
        "iou_overlap_threshold": iou_overlap_threshold,
        "severity_counts": severity_counts if len(all_flagged) > 0 else {},
        "reason_counts": reason_counts if len(all_flagged) > 0 else {},
        "flagging_rules": [
            {
                "rule": "low_confidence",
                "threshold": confidence_threshold,
                "severity": "MEDIUM",
            },
            {
                "rule": "overlapping_predictions",
                "threshold": iou_overlap_threshold,
                "severity": "MEDIUM",
            },
            {"rule": "ambiguous_class", "severity": "LOW"},
            {"rule": "large_bounding_box", "threshold": 0.4, "severity": "LOW"},
            {"rule": "edge_detection", "threshold": 0.05, "severity": "LOW"},
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")

    # Generate clinical review recommendations
    print("\n" + "=" * 60)
    print("CLINICAL REVIEW RECOMMENDATIONS")
    print("=" * 60)

    if len(all_flagged) > 0:
        high_priority = all_flagged[all_flagged["flag_severity"] == "HIGH"]
        medium_priority = all_flagged[all_flagged["flag_severity"] == "MEDIUM"]

        if len(high_priority) > 0:
            print(
                f"\n  [HIGH PRIORITY] {len(high_priority)} cases need immediate review:"
            )
            for _, row in high_priority.head(10).iterrows():
                print(
                    f"    - {row['image_name']}: {row['predicted_class']} "
                    f"(conf={row['confidence']:.3f}) [{row['flag_reason']}]"
                )

        if len(medium_priority) > 0:
            print(
                f"\n  [MEDIUM PRIORITY] {len(medium_priority)} cases should be reviewed:"
            )
            for _, row in medium_priority.head(5).iterrows():
                print(
                    f"    - {row['image_name']}: {row['predicted_class']} "
                    f"(conf={row['confidence']:.3f}) [{row['flag_reason']}]"
                )
    else:
        print("\n  All predictions passed uncertainty checks.")

    print("\n" + "=" * 60)
    print("Uncertainty flagging complete!")
    print("=" * 60)

    return all_flagged


def main():
    parser = argparse.ArgumentParser(
        description="Flag uncertain predictions for clinical review"
    )
    parser.add_argument("--input", type=str, default=None, help="Predictions CSV file")
    parser.add_argument(
        "--confidence", type=float, default=0.70, help="Confidence threshold"
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU overlap threshold")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    try:
        flagged = run_uncertainty_flagging(
            predictions_csv=args.input,
            confidence_threshold=args.confidence,
            iou_overlap_threshold=args.iou,
            output_dir=args.output,
        )
    except Exception as e:
        print(f"[ERROR] Uncertainty flagging failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
