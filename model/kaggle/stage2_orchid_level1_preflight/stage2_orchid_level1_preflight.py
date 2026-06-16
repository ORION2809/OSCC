"""Kaggle preflight runner for OralPath Level 1 Stage 2 training."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


KERNEL_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = KERNEL_ROOT / "bundle"
CONFIG_PATH = BUNDLE_ROOT / "model" / "training" / "stage2_grading" / "config.kaggle.yaml"
DATASET_CANDIDATES = [
    Path("/kaggle/input/orchid-oscc-classification"),
    Path("/kaggle/input/orchid-oscc-classification/ORCHID_train"),
]


def run(command: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def require_gpu() -> None:
    import torch

    print(f"[GPU] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle kernel started without GPU. Refusing to continue preflight.")


def configure_huggingface_token() -> None:
    if os.environ.get("HF_TOKEN"):
        print("[HF] HF_TOKEN already present in environment.", flush=True)
        return

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError(f"Kaggle secret HF_TOKEN is required and was not available: {exc}") from exc

    if not token:
        raise RuntimeError("Kaggle secret HF_TOKEN is empty.")

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    print("[HF] Hugging Face token loaded from Kaggle secret HF_TOKEN.", flush=True)


def find_orchid_root() -> Path:
    for candidate in DATASET_CANDIDATES:
        if (candidate / "ORCHID_train" / "train").exists():
            return candidate
        if (candidate / "train").exists():
            return candidate

    input_root = Path("/kaggle/input")
    for path in input_root.rglob("train"):
        if path.is_dir() and path.parent.name.lower().startswith("orchid"):
            return path.parent.parent if path.parent.name == "ORCHID_train" else path.parent

    raise FileNotFoundError("Could not find attached ORCHID Kaggle dataset under /kaggle/input")


def copy_outputs() -> None:
    candidates = [
        Path("/kaggle/working/stage2_logs/stage2_report.json"),
        Path("/kaggle/working/stage2_checkpoints/stage2_last.pt"),
        Path("/kaggle/working/stage2_checkpoints/stage2_best.pt"),
    ]
    for path in candidates:
        if path.exists():
            target = Path("/kaggle/working") / path.name
            shutil.copy2(path, target)
            print(f"[OUTPUT] {target}", flush=True)


def main() -> int:
    print("OralPath Kaggle Stage 2 preflight", flush=True)
    print(f"Python: {sys.version}", flush=True)
    if not BUNDLE_ROOT.exists():
        raise FileNotFoundError(f"Missing bundled code at {BUNDLE_ROOT}")

    require_gpu()
    configure_huggingface_token()

    orchid_root = find_orchid_root()
    os.environ["ORALPATH_ORCHID_ROOT"] = str(orchid_root)
    print(f"[DATA] ORALPATH_ORCHID_ROOT={orchid_root}", flush=True)

    run(
        [
            sys.executable,
            "model/training/stage2_grading/train.py",
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
        ],
        cwd=BUNDLE_ROOT,
    )

    run(
        [
            sys.executable,
            "model/training/stage2_grading/train.py",
            "--config",
            str(CONFIG_PATH),
            "--max-batches",
            "1",
            "--max-epochs",
            "1",
            "--state-output",
            "/kaggle/working/stage2_checkpoints/stage2_last.pt",
        ],
        cwd=BUNDLE_ROOT,
    )

    copy_outputs()
    print("[DONE] Stage 2 preflight complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
