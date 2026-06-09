"""Download/setup datasets on Colab's temporary /content disk.

Use this when Google Drive does not have enough space. The tradeoff is that
the data disappears when the Colab runtime is recycled, so downloads may need
to be repeated in a new session.

Stage 1 Kaggle OSCC is the recommended first target for ephemeral setup.
Stage 2 ORCHID is much larger and should be downloaded only when the runtime
has enough free disk and enough time left.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "model" / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "model" / "data" / "processed"

KAGGLE_SLUG = "ashenafifasilkebede/dataset"
ORCHID_FILES = {
    "train": ("https://zenodo.org/api/records/12636426/files/train.zip/content", RAW_ROOT / "train.zip"),
    "val": ("https://zenodo.org/api/records/12646943/files/val.zip/content", RAW_ROOT / "val.zip"),
    "test": ("https://zenodo.org/api/records/12646943/files/test.zip/content", RAW_ROOT / "test.zip"),
}


def run(command: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def print_disk() -> None:
    total, used, free = shutil.disk_usage(REPO_ROOT)
    print(
        "[DISK] total={:.1f} GB used={:.1f} GB free={:.1f} GB".format(
            total / 1e9,
            used / 1e9,
            free / 1e9,
        )
    )


def configure_kaggle_from_colab_secret() -> None:
    """Create ~/.kaggle/kaggle.json from Colab secrets if available."""
    try:
        from google.colab import userdata
    except Exception:
        return

    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    if not username or not key:
        return

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(
        '{{"username":"{}","key":"{}"}}'.format(username, key),
        encoding="utf-8",
    )
    kaggle_json.chmod(0o600)
    print("[OK] Kaggle credentials loaded from Colab secrets.")


def normalize_kaggle_tree(dataset_root: Path) -> None:
    """Normalize common Kaggle extraction layouts to train/val/test/Normal|OSCC."""
    # Already good.
    if (dataset_root / "train").exists() and (dataset_root / "test").exists():
        print("[OK] Kaggle tree already has train/test split.")
        return

    nested_candidates = [
        path
        for path in dataset_root.rglob("*")
        if path.is_dir() and (path / "100x").exists() or path.is_dir() and (path / "400x").exists()
    ]
    if nested_candidates:
        print("[INFO] Found raw 100x/400x layout. Leaving as-is for manual preprocessing.")
        return

    # Some Kaggle mirrors use validation instead of val.
    validation = dataset_root / "validation"
    if validation.exists() and not (dataset_root / "val").exists():
        validation.rename(dataset_root / "val")
        print("[OK] Renamed validation -> val")


def setup_kaggle_oscc() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    target = RAW_ROOT / "kaggle_oscc"
    if target.exists() and any(target.iterdir()):
        print(f"[SKIP] Kaggle OSCC already present: {target}")
        return

    configure_kaggle_from_colab_secret()
    run(["python", "-m", "pip", "install", "-q", "kaggle"])

    download_dir = RAW_ROOT / "kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(download_dir)])

    zip_files = list(download_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No Kaggle zip downloaded into {download_dir}")

    target.mkdir(parents=True, exist_ok=True)
    for archive in zip_files:
        print(f"[EXTRACT] {archive} -> {target}")
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(target)

    normalize_kaggle_tree(target)
    shutil.rmtree(download_dir, ignore_errors=True)
    print(f"[OK] Kaggle OSCC ready: {target}")


def download_file(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with dest.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)


def setup_orchid(download_only: bool = False) -> None:
    print("[WARN] ORCHID is large. Use this only if Colab has enough free disk.")
    print_disk()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for split, (url, archive) in ORCHID_FILES.items():
        if not archive.exists():
            print(f"[DOWNLOAD] ORCHID {split}: {url}")
            download_file(url, archive)
        else:
            print(f"[SKIP] Existing archive: {archive}")

        if download_only:
            continue

        run(["python", "scripts/extract_orchid.py"], cwd=REPO_ROOT)
        archive.unlink(missing_ok=True)
        print(f"[DELETE] Removed archive after extraction: {archive}")

    print(f"[OK] ORCHID processed path: {PROCESSED_ROOT / 'orchid'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup Colab ephemeral datasets")
    parser.add_argument("--stage", choices=["stage1", "stage2", "all"], default="stage1")
    parser.add_argument("--orchid-download-only", action="store_true")
    args = parser.parse_args()

    print_disk()
    if args.stage in ("stage1", "all"):
        setup_kaggle_oscc()
    if args.stage in ("stage2", "all"):
        setup_orchid(download_only=args.orchid_download_only)
    print_disk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
