"""Link Google Drive datasets into the Colab repo checkout.

Expected Drive layout:

    /content/drive/MyDrive/oralpath/data/
      kaggle_oscc/
        train/Normal/*.jpg
        train/OSCC/*.jpg
        val/Normal/*.jpg
        val/OSCC/*.jpg
        test/Normal/*.jpg
        test/OSCC/*.jpg
      orchid/
        train/normal/*.png
        train/osmf/*.png
        train/wdoscc/*.png
        train/mdoscc/*.png
        train/pdoscc/*.png
        val/...
        test/...

This script does not upload or download datasets. It links already-present
Drive folders into the repo paths used by the training configs.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_link_or_copy(source: Path, target: Path) -> None:
    """Make target point at source.

    Prefer symlinks because Colab supports them and large datasets should stay
    in Drive. If symlink creation fails, fall back to copying only as a last
    resort.
    """
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")

    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if target.is_dir() and any(target.iterdir()):
            print(f"[KEEP] Existing non-empty target: {target}")
            return
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
        print(f"[LINK] {target} -> {source}")
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        print(f"[COPY] {source} -> {target}")


def link_split_dirs(source_root: Path, target_root: Path) -> None:
    """Link train/val/test directories while preserving tracked manifests."""
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source not found: {source_root}")

    target_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        source_split = source_root / split
        target_split = target_root / split
        ensure_link_or_copy(source_split, target_split)


def main() -> int:
    parser = argparse.ArgumentParser(description="Link Drive datasets into Colab repo paths")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/oralpath/data",
        help="Drive data root containing kaggle_oscc/ and orchid/",
    )
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    print(f"[INFO] Drive data root: {drive_root}")

    kaggle_source = drive_root / "kaggle_oscc"
    orchid_source = drive_root / "orchid"

    kaggle_target = REPO_ROOT / "model" / "data" / "raw" / "kaggle_oscc"
    orchid_target = REPO_ROOT / "model" / "data" / "processed" / "orchid"

    missing = []
    if not kaggle_source.exists():
        missing.append(kaggle_source)
    if not orchid_source.exists():
        missing.append(orchid_source)

    if missing:
        print("[ERROR] Dataset folders are missing in Drive:")
        for path in missing:
            print(f"  - {path}")
        print("\nUpload or copy only the dataset folders to Drive, not the whole repo:")
        print("  - local model/data/raw/kaggle_oscc -> MyDrive/oralpath/data/kaggle_oscc")
        print("  - local model/data/processed/orchid -> MyDrive/oralpath/data/orchid")
        return 1

    ensure_link_or_copy(kaggle_source, kaggle_target)
    link_split_dirs(orchid_source, orchid_target)

    print("\n[OK] Dataset links are ready.")
    print("Next: python scripts/verify_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
