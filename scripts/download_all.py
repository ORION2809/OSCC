"""
OralPath — Download all models and datasets needed for the project.

Usage:
    source .venv/Scripts/activate
    python scripts/download_all.py

Requires:
    - HuggingFace token (already authenticated via hf auth login)
    - Internet connection
"""

import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path
from urllib.parse import urlparse

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_CACHE = PROJECT_ROOT / "model" / "external" / "weights"
DATA_ROOT = PROJECT_ROOT / "model" / "data" / "raw"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# HuggingFace Models
# ============================================================

def download_hf_model(repo_id: str, local_dir: Path, allow_patterns=None):
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"\n[HF] Downloading {repo_id} ...")
        print(f"       -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        print(f"[HF] OK {repo_id} downloaded successfully.")
        return True
    except Exception as e:
        print(f"[HF] FAIL Failed to download {repo_id}: {e}")
        return False


def download_hf_dataset(repo_id: str, local_dir: Path):
    """Download a dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"\n[HF Dataset] Downloading {repo_id} ...")
        print(f"       -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            repo_type="dataset",
            resume_download=True,
        )
        print(f"[HF Dataset] OK {repo_id} downloaded successfully.")
        return True
    except Exception as e:
        print(f"[HF Dataset] FAIL Failed to download {repo_id}: {e}")
        return False


# ============================================================
# External Downloads (HTTP)
# ============================================================

def download_file(url: str, dest: Path, chunk_size=8192):
    """Download a file with progress."""
    try:
        print(f"\n[HTTP] Downloading {url} ...")
        print(f"       -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r       Progress: {percent:.1f}%")
                        sys.stdout.flush()
        
        print(f"\n[HTTP] OK Downloaded: {dest.name} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n[HTTP] FAIL Failed: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar archive."""
    try:
        print(f"[Extract] Extracting {archive_path.name} ...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(dest_dir)
        elif archive_path.suffix in ('.tar', '.gz', '.tgz', '.bz2'):
            with tarfile.open(archive_path, 'r:*') as t:
                t.extractall(dest_dir)
        else:
            print(f"[Extract] Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"[Extract] OK Extracted to {dest_dir}")
        return True
    except Exception as e:
        print(f"[Extract] FAIL Failed: {e}")
        return False


# ============================================================
# Main Download Orchestrator
# ============================================================

def download_models():
    """Download all foundation model weights."""
    print("=" * 60)
    print("DOWNLOADING FOUNDATION MODELS")
    print("=" * 60)
    
    results = {}
    
    # 1. UNI (primary backbone candidate)
    results['uni'] = download_hf_model(
        repo_id="mahmoodlab/UNI",
        local_dir=MODEL_CACHE / "uni",
    )
    
    # 2. CONCH (benchmark comparison)
    results['conch'] = download_hf_model(
        repo_id="mahmoodlab/CONCH",
        local_dir=MODEL_CACHE / "conch",
    )
    
    # 3. CTransPath - academic model, not on HF. Skip for now.
    print("\n[INFO] CTransPath requires academic download from:")
    print("       https://github.com/Xiyue-Wang/TransPath")
    print("       Skipping — download manually if needed.")
    results['ctranspath'] = None
    
    return results


def download_datasets():
    """Download all datasets."""
    print("\n" + "=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)
    
    results = {}
    
    # 1. Kaggle OSCC
    # Requires Kaggle API credentials. Check if available.
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_creds.exists():
        print("\n[Kaggle] Kaggle credentials found. Attempting download...")
        try:
            import subprocess
            oscc_dir = DATA_ROOT / "kaggle_oscc"
            oscc_dir.mkdir(parents=True, exist_ok=True)
            # The exact dataset slug may vary -- common ones:
            # "srough/alternate-oral-cancer" or "train"
            # Let user know they may need to find the exact one
            print("[Kaggle] Attempting: kaggle datasets download -d train")
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", "train", "-p", str(oscc_dir)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print("[Kaggle] OK Kaggle OSCC downloaded.")
                results['kaggle_oscc'] = True
            else:
                print(f"[Kaggle] FAIL Failed: {result.stderr}")
                results['kaggle_oscc'] = False
        except Exception as e:
            print(f"[Kaggle] FAIL Error: {e}")
            results['kaggle_oscc'] = False
    else:
        print("\n[Kaggle] No Kaggle credentials found at ~/.kaggle/kaggle.json")
        print("         Please set up Kaggle API credentials:")
        print("         1. Go to https://www.kaggle.com/account")
        print("         2. Create new API token (downloads kaggle.json)")
        print("         3. Place it at ~/.kaggle/kaggle.json")
        print("         4. Then run this script again")
        print("\n         Alternative: manually download from:")
        print("         https://www.kaggle.com/datasets (search 'OSCC histopathological')")
        results['kaggle_oscc'] = False
    
    # 2. ORCHID Dataset
    # Try HuggingFace datasets first
    print("\n[ORCHID] Trying HuggingFace datasets...")
    orchid_hf = download_hf_dataset(
        repo_id="NishaChaudhary23/ORCHID",
        local_dir=DATA_ROOT / "orchid_hf",
    )
    if orchid_hf:
        results['orchid'] = True
    else:
        # Fallback: Zenodo
        print("\n[ORCHID] Trying Zenodo download...")
        print("         Zenodo record: https://zenodo.org/records/12636426")
        print("         Please download manually if HF dataset is not available.")
        results['orchid'] = False
    
    # 3. NDB-UFES (deferred to v1.2)
    print("\n[NDB-UFES] This dataset is deferred to v1.2 (segmentation).")
    print("           Skipping for now.")
    print("           Source: Academic repository (search 'NDB-UFES oral cancer')")
    results['ndb_ufes'] = None
    
    return results


def print_summary(model_results, dataset_results):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    print("\nModels:")
    for name, status in model_results.items():
        icon = "OK" if status else ("FAIL" if status is False else "SKIP")
        print(f"  {icon} {name}")
    
    print("\nDatasets:")
    for name, status in dataset_results.items():
        icon = "OK" if status else ("FAIL" if status is False else "SKIP")
        print(f"  {icon} {name}")
    
    print("\n" + "=" * 60)
    print("LOCAL PATHS")
    print("=" * 60)
    print(f"  Model weights:  {MODEL_CACHE}")
    print(f"  Datasets:       {DATA_ROOT}")
    
    # List what's actually there
    print("\n  Model cache contents:")
    if MODEL_CACHE.exists():
        for item in MODEL_CACHE.iterdir():
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            print(f"    - {item.name} ({size / 1024 / 1024:.1f} MB)")
    
    print("\n  Dataset contents:")
    if DATA_ROOT.exists():
        for item in DATA_ROOT.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                print(f"    - {item.name} ({size / 1024 / 1024:.1f} MB)")


def main():
    model_results = download_models()
    dataset_results = download_datasets()
    print_summary(model_results, dataset_results)


if __name__ == "__main__":
    main()
