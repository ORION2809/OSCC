"""
OralPath — Dataset Download Helper
Guided download instructions for all required datasets.

This script does NOT auto-download Kaggle or HuggingFace datasets
(requires authentication). Instead, it prints step-by-step instructions
and verifies local paths once downloaded.
"""

import argparse
from pathlib import Path

from model.data.preprocessing.dataset_loader import DatasetManifest


def print_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           OralPath — Dataset Download Instructions               ║
╠══════════════════════════════════════════════════════════════════╣

1. Kaggle OSCC Dataset
   ─────────────────────
   Source: https://www.kaggle.com/datasets (search "OSCC histopathological")
   License: CC BY 4.0
   Steps:
     a. Sign in to Kaggle
     b. Download the dataset
     c. Extract to: model/data/raw/kaggle_oscc/
     d. Expected structure:
        model/data/raw/kaggle_oscc/
        ├── normal/
        └── oscc/

2. ORCHID Dataset
   ───────────────
   Source: https://huggingface.co/datasets (search "ORCHID")
   License: CC BY 4.0
   Steps:
     a. Use huggingface-cli or download manually
     b. Extract to: model/data/raw/orchid/
     c. Expected structure:
        model/data/raw/orchid/
        ├── normal/
        ├── osmf/
        ├── wd/
        ├── md/
        └── pd/

3. OralPatho Repository
   ─────────────────────
   Source: GitHub (search "OralPatho OSCC")
   License: MIT
   Steps:
     a. Clone to model/external/oralpatho/
     b. Or run: git clone <url> model/external/oralpatho

4. Foundation Model Weights (optional for now)
   ───────────────────────────────────────────
   UNI:      huggingface-cli download mahmoodlab/UNI
   CONCH:    huggingface-cli download mahmoodlab/CONCH
   CTransPath: Follow academic repository instructions

5. NDB-UFES (deferred to v1.2)
   ────────────────────────────
   Do not download until segmentation workstream begins.

╚══════════════════════════════════════════════════════════════════╝
""")


def verify_local():
    """Verify which datasets are already present locally."""
    print("\n=== Local Dataset Verification ===\n")
    manifest_dir = Path("model/data/manifests")
    for manifest_file in manifest_dir.glob("*.json"):
        manifest = DatasetManifest(str(manifest_file))
        valid = manifest.validate()
        status = "✅ PRESENT" if valid else "❌ MISSING"
        print(f"{status} — {manifest.name} ({manifest.local_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify local dataset presence")
    args = parser.parse_args()

    if args.verify:
        verify_local()
    else:
        print_instructions()
        print("\n")
        verify_local()


if __name__ == "__main__":
    main()
