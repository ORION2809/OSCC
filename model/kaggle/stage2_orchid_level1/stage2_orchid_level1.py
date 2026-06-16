"""Kaggle background runner for OralPath Level 1 Stage 2 training.

Inputs:
    /kaggle/input/orchid-oscc-classification

Outputs:
    /kaggle/working/stage2_report.json
    /kaggle/working/stage2_last.pt
    /kaggle/working/stage2_best.pt
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/ORION2809/OSCC.git"
REPO_ROOT = Path("/kaggle/working/OSCC")
CONFIG_PATH = REPO_ROOT / "model" / "training" / "stage2_grading" / "config.kaggle.yaml"
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


def clone_or_update_repo() -> None:
    if REPO_ROOT.exists():
        run(["git", "fetch", "origin"], cwd=REPO_ROOT)
        run(["git", "checkout", "main"], cwd=REPO_ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=REPO_ROOT)
    else:
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)])


def install_dependencies() -> None:
    packages = [
        "timm>=0.9.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        "PyYAML>=6.0",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *packages])


def configure_huggingface_token() -> None:
    if os.environ.get("HF_TOKEN"):
        print("[HF] HF_TOKEN already present in environment.", flush=True)
        return

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        print(f"[HF] No Kaggle HF_TOKEN secret available: {exc}", flush=True)
        token = None

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("[HF] Hugging Face token loaded from Kaggle secret HF_TOKEN.", flush=True)
    else:
        print("[HF] UNI may fail if the model is gated. Add Kaggle secret HF_TOKEN if needed.", flush=True)


def copy_outputs() -> None:
    mappings = {
        REPO_ROOT / "model" / "training" / "stage2_grading" / "logs" / "stage2_report.json": Path(
            "/kaggle/working/stage2_report.json"
        ),
        Path("/kaggle/working/stage2_logs/stage2_report.json"): Path("/kaggle/working/stage2_report.json"),
        Path("/kaggle/working/stage2_checkpoints/stage2_last.pt"): Path("/kaggle/working/stage2_last.pt"),
        Path("/kaggle/working/stage2_checkpoints/stage2_best.pt"): Path("/kaggle/working/stage2_best.pt"),
    }

    for source, destination in mappings.items():
        if source.exists():
            shutil.copy2(source, destination)
            print(f"[OUTPUT] {destination} ({destination.stat().st_size} bytes)", flush=True)
        else:
            print(f"[WARN] Missing expected artifact: {source}", flush=True)


def main() -> int:
    print("OralPath Kaggle Stage 2 runner", flush=True)
    print(f"Python: {sys.version}", flush=True)
    run([sys.executable, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"])

    clone_or_update_repo()
    install_dependencies()
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
        cwd=REPO_ROOT,
    )

    run(
        [
            sys.executable,
            "model/training/stage2_grading/train.py",
            "--config",
            str(CONFIG_PATH),
            "--state-output",
            "/kaggle/working/stage2_checkpoints/stage2_last.pt",
        ],
        cwd=REPO_ROOT,
    )

    copy_outputs()
    print("[DONE] Stage 2 Kaggle training complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
