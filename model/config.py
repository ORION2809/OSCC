"""
Shared configuration for OralPath training pipeline.
Used by training, evaluation, export, and Android UI.
"""
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Dataset paths
KAGGLE_OSCC_PATH = ROOT / "model/data/raw/kaggle_oscc"
ORCHID_PATH = ROOT / "model/data/processed/orchid"

# Model weights paths
UNI_WEIGHTS = ROOT / "model/external/weights/uni"
CONCH_WEIGHTS = ROOT / "model/external/weights/conch"
CTRANS_PATH_WEIGHTS = ROOT / "model/external/weights/ctranspath/ctranspath.pth"

# Stage 1: Binary OSCC Detection
STAGE1 = {
    "task": "binary",
    "classes": ["Normal", "OSCC"],
    "class_to_idx": {"Normal": 0, "OSCC": 1},
    "dataset": KAGGLE_OSCC_PATH,
    "img_size": 224,
    "backbone": "uni",  # or "ctranspath", "efficientnet_b3"
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 50,
    "patience": 10,
    "target_sensitivity": 0.95,
    "target_specificity": 0.90,
}

# Stage 2: 5-Class Grading
STAGE2 = {
    "task": "multiclass",
    "classes": ["normal", "osmf", "wdoscc", "mdoscc", "pdoscc"],
    "class_labels": {
        "normal": "Normal oral mucosa",
        "osmf": "Oral Submucous Fibrosis",
        "wdoscc": "Well differentiated OSCC",
        "mdoscc": "Moderately differentiated OSCC",
        "pdoscc": "Poorly differentiated OSCC",
    },
    "class_to_idx": {"normal": 0, "osmf": 1, "wdoscc": 2, "mdoscc": 3, "pdoscc": 4},
    "dataset": ORCHID_PATH,
    "img_size": 224,
    "backbone": "uni",
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 50,
    "patience": 10,
    "target_accuracy": 0.85,
}

# Export
EXPORT = {
    "onnx": ROOT / "model/exports/oralpath_stage1.onnx",
    "tflite": ROOT / "model/exports/oralpath_stage1.tflite",
    "output_dir": ROOT / "model/exports",
}
