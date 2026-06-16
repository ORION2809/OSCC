"""Remote Colab CLI job for OralPath Stage 2 (Level 1 five-class grading).

This script is executed on the Colab VM via:

    colab exec -s oralpath-stage2 -f scripts/colab_cli_stage2_job.py

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
HF_TOKEN_UPLOAD = Path("/content/hf_token.secret")
FULL_STAGE2_FLAG = Path("/content/full_stage2.flag")
FULL_STAGE2_EPOCHS = Path("/content/full_stage2_epochs.txt")
STAGE2_RESUME_STATE = Path("/content/stage2_resume.pt")
STAGE2_CHUNK_MODE = Path("/content/stage2_chunk_mode.flag")
DEPS_MARKER = Path("/content/oralpath_deps_installed.flag")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


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


def configure_huggingface() -> None:
    if not HF_TOKEN_UPLOAD.exists():
        print("[INFO] No Hugging Face token uploaded; gated backbones may fall back.", flush=True)
        return

    token = HF_TOKEN_UPLOAD.read_text(encoding="utf-8").strip()
    if not token:
        print("[INFO] Uploaded Hugging Face token file is empty.", flush=True)
        return

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    print("[OK] Hugging Face token configured for gated model access.", flush=True)


def clone_or_update_repo() -> None:
    if REMOTE_REPO.exists():
        run(["git", "fetch", "origin"], cwd=REMOTE_REPO)
        run(["git", "checkout", "main"], cwd=REMOTE_REPO)
        run(["git", "reset", "--hard", "origin/main"], cwd=REMOTE_REPO)
    else:
        run(["git", "clone", REPO_URL, str(REMOTE_REPO)])


def install_dependencies() -> None:
    if DEPS_MARKER.exists():
        print("[SKIP] Dependencies already installed for this Colab session.", flush=True)
        return

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REMOTE_REPO)
    DEPS_MARKER.write_text("ok\n", encoding="utf-8")


def run_stage2() -> None:
    run([sys.executable, "scripts/colab_runtime_check.py"], cwd=REMOTE_REPO)
    run([sys.executable, "scripts/setup_colab_ephemeral_data.py", "--stage", "stage2"], cwd=REMOTE_REPO)
    run([sys.executable, "scripts/verify_datasets.py"], cwd=REMOTE_REPO)
    if STAGE2_CHUNK_MODE.exists():
        print("[SKIP] Chunk mode enabled; skipping dry-run and smoke train.", flush=True)
    else:
        run(
            [
                sys.executable,
                "model/training/stage2_grading/train.py",
                "--config",
                "model/training/stage2_grading/config.yaml",
                "--dry-run",
            ],
            cwd=REMOTE_REPO,
        )
        run(
            [
                sys.executable,
                "model/training/stage2_grading/train.py",
                "--config",
                "model/training/stage2_grading/config.yaml",
                "--max-batches",
                "1",
                "--max-epochs",
                "1",
            ],
            cwd=REMOTE_REPO,
        )

    if os.environ.get("ORALPATH_FULL_STAGE2") == "1" or FULL_STAGE2_FLAG.exists():
        command = [
            sys.executable,
            "model/training/stage2_grading/train.py",
            "--config",
            "model/training/stage2_grading/config.yaml",
        ]
        if FULL_STAGE2_EPOCHS.exists():
            epoch_limit = FULL_STAGE2_EPOCHS.read_text(encoding="utf-8").strip()
            if epoch_limit:
                command.extend(["--max-epochs", epoch_limit])
        if STAGE2_RESUME_STATE.exists():
            command.extend(["--resume-state", str(STAGE2_RESUME_STATE)])
        command.extend(
            [
                "--state-output",
                "model/training/stage2_grading/checkpoints/stage2_last.pt",
            ]
        )
        run(command, cwd=REMOTE_REPO)
    else:
        print("[SKIP] Full Stage 2 training skipped. Set ORALPATH_FULL_STAGE2=1 to run it.", flush=True)


def main() -> int:
    disk("start")
    configure_kaggle()
    configure_huggingface()
    clone_or_update_repo()
    install_dependencies()
    run_stage2()
    disk("end")
    print("[DONE] OralPath Stage 2 Colab CLI job completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
