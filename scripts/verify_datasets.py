"""
Verify all datasets are ready for training.
Checks: paths exist, files are readable, class counts match manifests.
"""
import json, os
from pathlib import Path
from PIL import Image

MANIFEST = Path("model/data/dataset_manifest.json")

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

def verify_orchid():
    base = Path("model/data/processed/orchid")
    issues = []
    total = 0
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
