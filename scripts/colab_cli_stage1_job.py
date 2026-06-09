"""Remote Colab CLI job for OralPath Stage 1.

This script is executed on the Colab VM via:

    colab exec -s oralpath-stage1 -f scripts/colab_cli_stage1_job.py

It assumes `/content/kaggle.json` has already been uploaded by the local
wrapper. It uses Colab's temporary disk, not Google Drive.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/ORION2809/OSCC.git"
REMOTE_REPO = Path("/content/oralpath")
KAGGLE_UPLOAD = Path("/content/kaggle.json")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def disk(label: str) -> None:
    total, used, free = shutil.disk_usage("/content")
    print(
        f"[DISK:{label}] total={total/1e9:.1f}GB used={used/1e9:.1f}GB free={free/1e9:.1f}GB",
        flush=True,
    )


def configure_kaggle() -> None:
    if not KAGGLE_UPLOAD.exists():
        raise FileNotFoundError(
            "/content/kaggle.json missing. Upload it before running this job."
        )
    data = json.loads(KAGGLE_UPLOAD.read_text(encoding="utf-8"))
    if not data.get("username") or not data.get("key"):
        raise ValueError("kaggle.json must contain both username and key.")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = kaggle_dir / "kaggle.json"
    target.write_text(json.dumps({"username": data["username"], "key": data["key"]}), encoding="utf-8")
    target.chmod(0o600)
    print(f"[OK] Kaggle credentials configured for {data['username']}", flush=True)


def clone_or_update_repo() -> None:
    if REMOTE_REPO.exists():
        run(["git", "fetch", "origin"], cwd=REMOTE_REPO)
        run(["git", "checkout", "main"], cwd=REMOTE_REPO)
        run(["git", "reset", "--hard", "origin/main"], cwd=REMOTE_REPO)
    else:
        run(["git", "clone", REPO_URL, str(REMOTE_REPO)])


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REMOTE_REPO)


def run_stage1() -> None:
    run([sys.executable, "scripts/colab_runtime_check.py"], cwd=REMOTE_REPO)
    run([sys.executable, "scripts/setup_colab_ephemeral_data.py", "--stage", "stage1"], cwd=REMOTE_REPO)
    run([sys.executable, "scripts/verify_datasets.py"], cwd=REMOTE_REPO)
    run(
        [
            sys.executable,
            "model/training/stage1_detection/train.py",
            "--config",
            "model/training/stage1_detection/config.yaml",
            "--dry-run",
        ],
        cwd=REMOTE_REPO,
    )
    run(
        [
            sys.executable,
            "model/training/stage1_detection/train.py",
            "--config",
            "model/training/stage1_detection/config.yaml",
            "--max-batches",
            "1",
        ],
        cwd=REMOTE_REPO,
    )

    if os.environ.get("ORALPATH_FULL_STAGE1") == "1":
        run(
            [
                sys.executable,
                "model/training/stage1_detection/train.py",
                "--config",
                "model/training/stage1_detection/config.yaml",
            ],
            cwd=REMOTE_REPO,
        )
    else:
        print("[SKIP] Full Stage 1 training skipped. Set ORALPATH_FULL_STAGE1=1 to run it.", flush=True)


def main() -> int:
    disk("start")
    configure_kaggle()
    clone_or_update_repo()
    install_dependencies()
    run_stage1()
    disk("end")
    print("[DONE] OralPath Stage 1 Colab CLI job completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
