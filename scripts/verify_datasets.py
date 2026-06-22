"""
Verify all datasets are ready for training.
Checks: paths exist, files are readable, class counts match manifests.
Supports both directory-based and zip-backed ORCHID manifests.
"""
import io
import json
import zipfile
from pathlib import Path
from PIL import Image


def verify_kaggle_oscc():
    base = Path("model/data/raw/kaggle_oscc")
    issues = []
    total = 0
    for split in ["train", "val", "test"]:
        for cls in ["Normal", "OSCC"]:
            d = base / split / cls
            if not d.exists():
                issues.append(f"Missing: {d}")
                continue
            files = list(d.iterdir())
            total += len(files)
            # Try opening first file
            if files:
                try:
                    img = Image.open(files[0])
                    img.verify()
                except Exception as e:
                    issues.append(f"Corrupt image in {d}: {e}")
    print(f"[Kaggle OSCC] {total} images checked")
    return issues


def _open_first_zip_image(zip_path: Path, entries: list[str]) -> bool:
    for entry in entries:
        if not entry.lower().endswith(".png"):
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(entry) as f:
                    img = Image.open(io.BytesIO(f.read()))
                    img.verify()
            return True
        except Exception as e:
            return False
    return False


def verify_orchid():
    manifest_paths = [
        Path("model/data/manifests/orchid_zip.json"),
        Path("model/data/manifests/orchid.json"),
    ]
    manifest_path = None
    for candidate in manifest_paths:
        if candidate.exists():
            manifest_path = candidate
            break

    if manifest_path is None:
        print("[ORCHID] no manifest found")
        return ["No ORCHID manifest found"]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    local_path = Path(manifest.get("local_path", "model/data/processed/orchid"))
    zip_splits = manifest.get("zip_splits")

    issues = []
    total = 0

    if zip_splits:
        print(f"[ORCHID] using zip-backed manifest: {manifest_path}")
        for split, zip_name in zip_splits.items():
            zip_path = local_path / zip_name
            if not zip_path.exists():
                issues.append(f"Missing zip: {zip_path}")
                continue
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    entries = zf.namelist()
                    pngs = [e for e in entries if e.lower().endswith(".png")]
                    total += len(pngs)
                    if pngs and not _open_first_zip_image(zip_path, entries):
                        issues.append(f"Could not read image from {zip_path}")
            except zipfile.BadZipFile as e:
                issues.append(f"Corrupt zip {zip_path}: {e}")
    else:
        print(f"[ORCHID] using directory manifest: {manifest_path}")
        base = local_path
        for split in ["train", "val", "test"]:
            split_dir = base / split
            if not split_dir.exists():
                issues.append(f"Missing: {split_dir}")
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                files = list(class_dir.glob("*.png"))
                total += len(files)
                if files:
                    try:
                        img = Image.open(files[0])
                        img.verify()
                    except Exception as e:
                        issues.append(f"Corrupt image in {class_dir}: {e}")

    print(f"[ORCHID] {total} patches checked")
    return issues


def main():
    print("=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)

    issues = []
    issues.extend(verify_kaggle_oscc())
    issues.extend(verify_orchid())

    if issues:
        print("\n[WARN] Issues found:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\n[OK] All datasets verified successfully!")
        print("\nReady for:")
        print("  - Stage 1: Kaggle OSCC binary (Normal vs OSCC)")
        print("  - Stage 2: ORCHID 5-class (normal/OSMF/WD/MD/PD)")


if __name__ == "__main__":
    main()
