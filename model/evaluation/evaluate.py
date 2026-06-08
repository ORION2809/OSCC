"""
OralPath — Evaluation Harness
Shared evaluation utilities for all model stages.

Metrics computed:
    - Accuracy, Precision, Recall, F1
    - AUC-ROC
    - Confusion matrix
    - Per-class metrics (for multi-class)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.95,
    target_specificity: float = 0.90,
) -> Dict:
    """Evaluate binary classification and return metrics dict."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "targets_met": {
            "sensitivity": bool(sensitivity >= target_sensitivity),
            "specificity": bool(specificity >= target_specificity),
        },
    }
    return metrics


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Evaluate multi-class classification and return metrics dict."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    for idx, name in enumerate(class_names):
        mask = y_true == idx
        metrics["per_class"][name] = {
            "precision": float(precision_score(y_true == idx, y_pred == idx, zero_division=0)),
            "recall": float(recall_score(y_true == idx, y_pred == idx, zero_division=0)),
            "f1": float(f1_score(y_true == idx, y_pred == idx, zero_division=0)),
        }

    return metrics


def print_report(metrics: Dict, title: str = "Evaluation Report") -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(json.dumps(metrics, indent=2))
    print("=" * 50)


def save_report(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Report saved to {output_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, size=n)
    y_prob = np.random.rand(n)
    y_pred = (y_prob > 0.5).astype(int)

    report = evaluate_binary(y_true, y_pred, y_prob)
    print_report(report, "Stage 1 (Binary) — Dummy Data")
    save_report(report, "model/evaluation/reports/dummy_stage1.json")
