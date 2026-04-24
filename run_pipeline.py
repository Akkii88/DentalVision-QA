"""
Main Pipeline Runner for DentalVision-QA
Orchestrates the complete ML pipeline.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent


def run_command(cmd, description, cwd=None):
    """Run a command and display output."""
    print(f"\n{'=' * 60}")
    print(f"▶ {description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}")
    print(f"Working Dir: {cwd or PROJECT_ROOT}")
    print()

    # Use python3 explicitly
    if cmd.startswith("python "):
        cmd = "python3" + cmd[6:]
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or PROJECT_ROOT, capture_output=False, text=True
    )

    if result.returncode != 0:
        print(f"\n[WARN] Command exited with code {result.returncode}")
        return result.returncode

    print(f"\n[SUCCESS] {description} complete")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run complete DentalVision-QA pipeline"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training (use existing model)",
    )
    parser.add_argument(
        "--skip-dataset", action="store_true", help="Skip dataset download"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10 for quick testing)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=480, help="Image size for training (default: 480)"
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   DENTALVISION-QA - END-TO-END PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skip Train: {args.skip_train}")
    print(f"Skip Dataset: {args.skip_dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Image Size: {args.imgsz}")
    print(f"Batch Size: {args.batch}")
    print("=" * 60)

    steps = []

    # Step 1: Download dataset
    if not args.skip_dataset:
        print("\n📦 STEP 1: Download Dataset")
        code = run_command(
            f"python {PROJECT_ROOT}/src/download_dataset.py",
            "Downloading dental datasets",
            cwd=PROJECT_ROOT,
        )
        steps.append(("Download Dataset", code))
    else:
        print("\n⏭️  Skipping dataset download")
        steps.append(("Download Dataset", "skipped"))

    # Step 2: Prepare dataset
    print("\n📂 STEP 2: Prepare Dataset")
    code = run_command(
        f"python {PROJECT_ROOT}/src/prepare_dataset.py",
        "Preparing dataset in YOLO format",
        cwd=PROJECT_ROOT,
    )
    steps.append(("Prepare Dataset", code))

    # Step 3: Train model
    if not args.skip_train:
        print("\n🤖 STEP 3: Train YOLO Model")
        code = run_command(
            f"python {PROJECT_ROOT}/src/train_yolo.py "
            f"--epochs {args.epochs} "
            f"--imgsz {args.imgsz} "
            f"--batch {args.batch}",
            "Training YOLOv8 model",
            cwd=PROJECT_ROOT,
        )
        steps.append(("Train Model", code))
    else:
        print("\n⏭️  Skipping model training")
        steps.append(("Train Model", "skipped"))

    # Step 4: Evaluate model
    print("\n📊 STEP 4: Evaluate Model")
    code = run_command(
        f"python {PROJECT_ROOT}/src/evaluate_model.py",
        "Evaluating model performance",
        cwd=PROJECT_ROOT,
    )
    steps.append(("Evaluate Model", code))

    # Step 5: Run predictions
    print("\n🎯 STEP 5: Run Predictions")
    code = run_command(
        f"python {PROJECT_ROOT}/src/predict.py",
        "Generating predictions",
        cwd=PROJECT_ROOT,
    )
    steps.append(("Run Predictions", code))

    # Step 6: QA/QC
    print("\n✅ STEP 6: QA/QC Assessment")
    code = run_command(
        f"python {PROJECT_ROOT}/src/qa_qc.py",
        "Running QA/QC assessment",
        cwd=PROJECT_ROOT,
    )
    steps.append(("QA/QC", code))

    # Step 7: Uncertainty flagging
    print("\n🚩 STEP 7: Uncertainty Flagging")
    code = run_command(
        f"python {PROJECT_ROOT}/src/uncertainty_flagging.py",
        "Flagging uncertain cases",
        cwd=PROJECT_ROOT,
    )
    steps.append(("Uncertainty Flagging", code))

    # Step 8: Annotation analytics
    print("\n📈 STEP 8: Annotation Analytics")
    code = run_command(
        f"python {PROJECT_ROOT}/src/annotation_analytics.py",
        "Generating analytics",
        cwd=PROJECT_ROOT,
    )
    steps.append(("Analytics", code))

    # Summary
    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for step_name, status in steps:
        status_str = "✓" if status == 0 else "⚠" if status == "skipped" else "✗"
        print(f"  {status_str} {step_name}")

    print("\n📁 Output Directories:")
    print(f"  - Models:       {PROJECT_ROOT}/outputs/models/")
    print(f"  - Predictions:  {PROJECT_ROOT}/outputs/predictions/")
    print(f"  - Metrics:      {PROJECT_ROOT}/outputs/metrics/")
    print(f"  - Reports:      {PROJECT_ROOT}/outputs/reports/")

    print("\n🚀 To launch dashboard:")
    print(f"  streamlit run {PROJECT_ROOT}/dashboard/app.py")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
