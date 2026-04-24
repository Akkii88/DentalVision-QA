"""
YOLO Training Module for DentalVision-QA
Trains YOLOv8 model on dental finding detection.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

try:
    import torch
except ImportError:
    print("Installing torch...")
    os.system("pip install torch torchvision")
    import torch

# ─── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "models"


def get_device():
    """Determine the best device for training."""
    # Check for MPS (Mac M2)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Using MPS (Apple Silicon)")
        return "mps"

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"

    print("[INFO] Using CPU")
    return "cpu"


def train_yolo(
    data_yaml=None,
    model_name="yolov8n.pt",
    epochs=30,
    imgsz=640,
    batch=16,
    device=None,
    project_dir=None,
    name="dental_detector",
):
    """Train YOLOv8 model on dental dataset."""

    if data_yaml is None:
        data_yaml = DATA_DIR / "data.yaml"

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if project_dir is None:
        project_dir = MODEL_DIR

    project_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device is None:
        device = get_device()

    print("=" * 60)
    print("DentalVision-QA YOLO Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"Data: {data_yaml}")
    print(f"Project: {project_dir / name}")
    print("=" * 60)

    # Load model
    print(f"\n[INFO] Loading model {model_name}...")
    model = YOLO(model_name)

    # Train
    print(f"\n[INFO] Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        patience=10,  # Early stopping patience
        save=True,
        save_period=-1,  # Save checkpoint every N epochs (-1 = disabled)
        cache=False,
        workers=4,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        resume=False,
        amp=True,  # Automatic Mixed Precision
    )

    # Get best model path
    best_model_path = project_dir / name / "weights" / "best.pt"
    last_model_path = project_dir / name / "weights" / "last.pt"

    print(f"\n[SUCCESS] Training complete!")
    if best_model_path.exists():
        print(f"  Best model: {best_model_path}")
    if last_model_path.exists():
        print(f"  Last model: {last_model_path}")

    # Save training config
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "data_yaml": str(data_yaml),
        "best_model": str(best_model_path) if best_model_path.exists() else None,
        "last_model": str(last_model_path) if last_model_path.exists() else None,
    }

    config_path = project_dir / name / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")

    return results, model


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for dental detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda, cpu, mps, or integer)"
    )
    parser.add_argument("--name", type=str, default="dental_detector", help="Run name")
    parser.add_argument("--project", type=str, default=None, help="Project directory")
    args = parser.parse_args()

    if args.data is None:
        data_yaml = DATA_DIR / "data.yaml"
    else:
        data_yaml = Path(args.data)

    if args.project is None:
        project_dir = MODEL_DIR
    else:
        project_dir = Path(args.project)

    try:
        results, model = train_yolo(
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project_dir=project_dir,
            name=args.name,
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
