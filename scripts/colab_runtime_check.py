"""Validate that a Colab runtime is ready for OralPath training.

Run this inside a notebook connected to the Google Colab VS Code extension:

    python scripts/colab_runtime_check.py

It checks runtime basics only. It does not login to Google, Hugging Face, or
Kaggle, and it does not upload project files to Drive.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("OralPath Colab runtime check")
    print("=" * 32)

    in_colab = has_module("google.colab")
    print(f"Colab module available: {in_colab}")

    if has_module("torch"):
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch: missing")

    repo_root = Path.cwd()
    expected = [
        repo_root / "requirements.txt",
        repo_root / "model" / "training" / "stage1_detection" / "train.py",
        repo_root / "model" / "data" / "manifests" / "kaggle_oscc.json",
    ]
    for path in expected:
        print(f"{path.relative_to(repo_root)}: {'OK' if path.exists() else 'missing'}")

    drive_root = Path("/content/drive/MyDrive/oralpath")
    if in_colab:
        print(f"Drive workspace exists: {drive_root.exists()} ({drive_root})")

    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN env var: set")
    else:
        print("HF_TOKEN env var: not set; use huggingface_hub.login() or Colab secrets when needed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
