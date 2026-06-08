"""
Extract and flatten ORCHID train/val/test zips for patch-level classification.
ORCHID structure: {split}/{class}/{wsi_id}/{patch_*.png}
Flattened output: processed/orchid/{split}/{class}/{wsi_id}_{patch_*.png}
"""
import zipfile, os, json, shutil
from pathlib import Path

RAW_DIR = Path("model/data/raw")
EXTRACT_DIR = Path("model/data/processed/orchid")

def extract_zip(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()
        print("[EXTRACT] {}: {} entries -> {}".format(zip_path.name, len(namelist), dest_dir))
        z.extractall(dest_dir)
    print("[OK] Extracted to {}".format(dest_dir))

def flatten_patches(src_dir, split_name):
    """
    Flatten WSI-based patches into single class directories.
    Input:  raw/orchid_{split}/{split}/{class}/{wsi_id}/{patch.png}
    Output: processed/orchid/{split}/{class}/{wsi_id}_{patch.png}
    """
    split_src = src_dir / split_name
    split_dst = EXTRACT_DIR / split_name
    
    if not split_src.exists():
        print("[SKIP] {} not found".format(split_src))
        return
    
    os.makedirs(split_dst, exist_ok=True)
    class_counts = {}
    
    for class_dir in sorted(split_src.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        class_dst = split_dst / class_name
        os.makedirs(class_dst, exist_ok=True)
        
        count = 0
        for wsi_dir in class_dir.iterdir():
            if not wsi_dir.is_dir():
                continue
            for img in wsi_dir.glob("*.png"):
                new_name = "{}_{}".format(wsi_dir.name, img.name)
                shutil.copy2(img, class_dst / new_name)
                count += 1
        
        class_counts[class_name] = count
        print("  {}: {} patches".format(class_name, count))
    
    manifest = {
        "split": split_name,
        "classes": class_counts,
        "total": sum(class_counts.values())
    }
    manifest_path = EXTRACT_DIR / "{}_manifest.json".format(split_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("[OK] {} manifest saved: {}".format(split_name, manifest_path))

def main():
    for zip_name, split_name in [("train.zip", "train"), ("val.zip", "val"), ("test.zip", "test")]:
        zip_path = RAW_DIR / zip_name
        extract_dir = RAW_DIR / "orchid_{}".format(split_name)
        
        if zip_path.exists():
            try:
                extract_zip(zip_path, extract_dir)
                flatten_patches(extract_dir, split_name)
            except zipfile.BadZipFile:
                print("[SKIP] {} is corrupt".format(zip_path))
        else:
            print("[SKIP] {} not found".format(zip_path))
    
    print("\n[SUMMARY] ORCHID dataset:")
    for split in ["train", "val", "test"]:
        manifest_path = EXTRACT_DIR / "{}_manifest.json".format(split)
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
            print("  {}: {} patches across {} classes".format(
                split, data["total"], len(data["classes"])))

if __name__ == "__main__":
    main()
