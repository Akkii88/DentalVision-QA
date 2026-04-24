"""
Annotation Analytics Module for DentalVision-QA
Generates charts and analysis for annotation quality.
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
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["axes.labelsize"] = 11
    matplotlib.rcParams["axes.titlesize"] = 13
except ImportError:
    print("Installing matplotlib...")
    os.system("pip install matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_style("whitegrid")
except ImportError:
    print("Installing seaborn...")
    os.system("pip install seaborn")
    import seaborn as sns

    sns.set_style("whitegrid")

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CLASS_NAMES = [
    "caries",
    "plaque",
    "calculus",
    "gingivitis",
    "missing_tooth",
    "filling_crown",
    "ambiguous",
]


def set_style():
    """Set consistent plot styling."""
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def plot_label_distribution(labels_dir=None, output_dir=None):
    """Plot distribution of labels per class."""
    if output_dir is None:
        output_dir = REPORTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if labels_dir is None:
        labels_dir = DATA_DIR / "labels"
    labels_dir = Path(labels_dir)

    class_counts = {cls: 0 for cls in CLASS_NAMES}

    for split in ["train", "val", "test"]:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        for lbl_file in split_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id < len(CLASS_NAMES):
                                class_counts[CLASS_NAMES[cls_id]] += 1

    # Also check predictions if available
    pred_csv = PREDICTIONS_DIR / "predictions.csv"
    if pred_csv.exists():
        df = pd.read_csv(pred_csv)
        for cls in CLASS_NAMES:
            class_counts[cls] += len(df[df["predicted_class"] == cls])

    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#A0A0A0",
    ]

    # Absolute counts
    ax1.bar(
        range(len(CLASS_NAMES)),
        [class_counts[c] for c in CLASS_NAMES],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(range(len(CLASS_NAMES)))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax1.set_ylabel("Number of Annotations")
    ax1.set_title("Label Distribution (Absolute Counts)")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (cls, count) in enumerate(class_counts.items()):
        ax1.text(
            i,
            count + max(class_counts.values()) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Pie chart
    total = sum(class_counts.values())
    if total > 0:
        wedges, texts, autotexts = ax2.pie(
            [class_counts[c] for c in CLASS_NAMES],
            labels=CLASS_NAMES,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title("Label Distribution (Percentage)")
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

    plt.suptitle("Dental Findings - Label Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "label_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved label distribution plot: {output_path}")

    return class_counts


def plot_confidence_distribution(predictions_csv=None, output_dir=None):
    """Plot distribution of prediction confidence scores."""
    if output_dir is None:
        output_dir = REPORTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if predictions_csv is None:
        predictions_csv = PREDICTIONS_DIR / "predictions.csv"
    predictions_csv = Path(predictions_csv)

    if not predictions_csv.exists():
        print("[WARN] Predictions CSV not found. Skipping confidence plot.")
        return {}

    df = pd.read_csv(predictions_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = [
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#FF6B6B",
        "#DDA0DD",
        "#A0A0A0",
    ]

    # 1. Overall histogram
    ax = axes[0, 0]
    ax.hist(df["confidence"], bins=30, color="#4ECDC4", edgecolor="black", alpha=0.7)
    ax.axvline(x=0.7, color="red", linestyle="--", label="Review Threshold (0.70)")
    ax.axvline(x=0.5, color="orange", linestyle=":", label="Warning Threshold (0.50)")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Confidence Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. By class
    ax = axes[0, 1]
    class_data = [
        df[df["predicted_class"] == cls]["confidence"].values
        for cls in CLASS_NAMES
        if len(df[df["predicted_class"] == cls]) > 0
    ]
    class_labels = [
        cls for cls in CLASS_NAMES if len(df[df["predicted_class"] == cls]) > 0
    ]
    if class_data:
        bp = ax.boxplot(class_data, labels=class_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(class_labels)]):
            patch.set_facecolor(color)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Confidence Distribution by Class")
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # 3. Confidence vs Box size scatter
    ax = axes[1, 0]
    if "x1" in df.columns and "x2" in df.columns:
        df["box_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
        for i, cls in enumerate(CLASS_NAMES):
            cls_df = df[df["predicted_class"] == cls]
            if len(cls_df) > 0:
                ax.scatter(
                    cls_df["box_area"],
                    cls_df["confidence"],
                    label=cls,
                    color=colors[i],
                    alpha=0.6,
                    s=30,
                )
        ax.set_xlabel("Bounding Box Area (pixels²)")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Confidence vs Bounding Box Size")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_conf = np.sort(df["confidence"].values)
    yvals = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
    ax.plot(sorted_conf, yvals, color="#FF6B6B", linewidth=2)
    ax.axvline(x=0.7, color="red", linestyle="--", label="0.70 threshold")
    ax.axvline(x=0.5, color="orange", linestyle=":", label="0.50 threshold")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Prediction Confidence Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "confidence_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confidence distribution plot: {output_path}")

    # Calculate statistics
    stats = {
        "mean_confidence": round(df["confidence"].mean(), 4),
        "median_confidence": round(df["confidence"].median(), 4),
        "std_confidence": round(df["confidence"].std(), 4),
        "min_confidence": round(df["confidence"].min(), 4),
        "max_confidence": round(df["confidence"].max(), 4),
        "below_05": int(len(df[df["confidence"] < 0.5])),
        "below_07": int(len(df[df["confidence"] < 0.7])),
        "above_09": int(len(df[df["confidence"] > 0.9])),
    }

    return stats


def plot_review_vs_pass(output_dir=None):
    """Plot review vs pass counts from QA results."""
    if output_dir is None:
        output_dir = REPORTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    qa_report = METRICS_DIR / "qa_report.csv"

    if not qa_report.exists():
        print("[WARN] QA report not found. Skipping review/pass plot.")
        return None

    df = pd.read_csv(qa_report)

    status_counts = df["status"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_pie = ["#2ECC71", "#F39C12", "#E74C3C"]
    colors_bar = ["#27AE60", "#F1C40F", "#C0392B"]

    # Pie chart
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        colors=colors_pie[: len(status_counts)],
        startangle=90,
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax.set_title("Review vs Pass Distribution")

    # Bar chart by class
    ax = axes[1]
    class_status = pd.crosstab(df["predicted_class"], df["status"])
    class_status.plot(kind="bar", stacked=True, ax=ax, color=colors_bar)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Review/Pass by Class")
    ax.legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("QA/QC Review Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "review_pass_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved review/pass plot: {output_path}")

    return status_counts.to_dict()


def plot_class_errors(output_dir=None):
    """Plot class-wise error analysis."""
    if output_dir is None:
        output_dir = REPORTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    qa_report = METRICS_DIR / "qa_report.csv"

    if not qa_report.exists():
        print("[WARN] QA report not found. Skipping error analysis plot.")
        return None

    df = pd.read_csv(qa_report)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#A0A0A0",
    ]

    # 1. Error rate by class
    ax = axes[0, 0]
    error_rates = []
    class_labels = []
    for cls in CLASS_NAMES:
        cls_data = df[df["predicted_class"] == cls]
        if len(cls_data) > 0:
            error_rate = len(cls_data[cls_data["status"] != "PASS"]) / len(cls_data)
            error_rates.append(error_rate * 100)
            class_labels.append(cls)

    ax.bar(
        range(len(error_rates)),
        error_rates,
        color=colors[: len(error_rates)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rate by Class")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. IoU distribution by class
    ax = axes[0, 1]
    class_iou_data = [
        df[df["predicted_class"] == cls]["iou"].dropna().values
        for cls in CLASS_NAMES
        if len(df[df["predicted_class"] == cls]) > 0
    ]
    class_iou_labels = [
        cls for cls in CLASS_NAMES if len(df[df["predicted_class"] == cls]) > 0
    ]
    if class_iou_data:
        bp = ax.boxplot(class_iou_data, labels=class_iou_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(class_iou_labels)]):
            patch.set_facecolor(color)
    ax.set_ylim([0, 1])
    ax.set_ylabel("IoU Score")
    ax.set_title("IoU Distribution by Class")
    ax.grid(axis="y", alpha=0.3)

    # 3. Confusion matrix (simplified)
    ax = axes[1, 0]
    # Count REVIEW vs MISSING_LABEL by class
    review_counts = []
    missing_counts = []
    for cls in class_labels:
        cls_data = df[df["predicted_class"] == cls]
        review_counts.append(len(cls_data[cls_data["status"] == "REVIEW"]))
        missing_counts.append(len(cls_data[cls_data["status"] == "MISSING_LABEL"]))

    x = range(len(class_labels))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        review_counts,
        width,
        label="Review",
        color="#F39C12",
    )
    ax.bar(
        [i + width / 2 for i in x],
        missing_counts,
        width,
        label="Missing",
        color="#E74C3C",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Review vs Missing by Class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Confidence of errors vs correct
    ax = axes[1, 1]
    correct_conf = df[df["status"] == "PASS"]["confidence"]
    error_conf = df[df["status"] != "PASS"]["confidence"]

    ax.hist(
        [correct_conf, error_conf],
        bins=20,
        label=["Correct", "Error"],
        color=["#27AE60", "#E74C3C"],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence: Correct vs Error Predictions")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Class-wise Error Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "class_errors.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved class errors plot: {output_path}")

    return {}


def generate_report(output_dir=None):
    """Generate comprehensive analytics report."""
    if output_dir is None:
        output_dir = REPORTS_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DentalVision-QA Annotation Analytics")
    print("=" * 60)

    set_style()

    # Generate all plots
    print("\n[INFO] Generating label distribution plot...")
    label_dist = plot_label_distribution(output_dir=output_dir)

    print("[INFO] Generating confidence distribution plot...")
    conf_stats = plot_confidence_distribution(output_dir=output_dir)

    print("[INFO] Generating review/pass distribution plot...")
    review_stats = plot_review_vs_pass(output_dir=output_dir)

    print("[INFO] Generating class error analysis plot...")
    error_stats = plot_class_errors(output_dir=output_dir)

    # Compile summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "label_distribution": label_dist,
        "confidence_stats": conf_stats,
        "review_stats": review_stats,
        "plots_generated": [
            "label_distribution.png",
            "confidence_distribution.png",
            "review_pass_distribution.png",
            "class_errors.png",
        ],
    }

    summary_path = output_dir / "analytics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUCCESS] Analytics summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Analytics generation complete!")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate annotation analytics")
    parser.add_argument("--labels", type=str, default=None, help="Label directory")
    parser.add_argument("--predictions", type=str, default=None, help="Predictions CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    try:
        generate_report(output_dir=args.output)
    except Exception as e:
        print(f"[ERROR] Analytics generation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
