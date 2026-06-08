"""
OralPath — Dataset Loader
Shared dataset loading utilities for Kaggle OSCC and ORCHID datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


class DatasetManifest:
    """Loads and validates a dataset manifest JSON."""

    def __init__(self, manifest_path: str):
        with open(manifest_path, "r") as f:
            self.config = json.load(f)
        self.name = self.config["name"]
        self.local_path = Path(self.config["local_path"])
        self.classes = self.config["classes"]
        self.class_aliases = self.config.get("class_aliases", {})
        self.split_ratio = self.config["split"]
        self.seed = self.config["seed"]

    def _resolve_class_dir(self, base_dir: Path, cls_name: str) -> Path | None:
        """Resolve class folders case-insensitively.

        Kaggle/ORCHID sources are not consistent about label casing
        (`normal` vs `Normal`, `oscc` vs `OSCC`). Keep manifests canonical
        and adapt to the downloaded folder names here.
        """
        candidates = [cls_name, *self.class_aliases.get(cls_name, [])]
        for candidate in candidates:
            exact = base_dir / candidate
            if exact.exists():
                return exact

        expected = {candidate.lower() for candidate in candidates}
        for child in base_dir.iterdir() if base_dir.exists() else []:
            if child.is_dir() and child.name.lower() in expected:
                return child
        return None

    def _load_samples_from_dir(self, base_dir: Path) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for cls_name, label_idx in self.classes.items():
            cls_dir = self._resolve_class_dir(base_dir, cls_name)
            if cls_dir is None:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                for fp in cls_dir.glob(ext):
                    samples.append((str(fp), label_idx))
        return samples

    def has_explicit_splits(self) -> bool:
        """Return true when dataset is already organized as train/val/test."""
        return all((self.local_path / split).exists() for split in ("train", "val", "test"))

    def validate(self) -> bool:
        """Check that expected directories and files exist."""
        if not self.local_path.exists():
            print(f"[ERROR] Dataset path does not exist: {self.local_path}")
            return False

        bases = [self.local_path]
        if self.has_explicit_splits():
            bases = [self.local_path / split for split in ("train", "val", "test")]

        valid = True
        for base_dir in bases:
            for cls_name in self.classes:
                cls_dir = self._resolve_class_dir(base_dir, cls_name)
                if cls_dir is None:
                    print(f"[ERROR] Class directory missing under {base_dir}: {cls_name}")
                    valid = False
                    continue
                files = list(cls_dir.glob("*"))
                if not files:
                    print(f"[WARNING] No files in class directory: {cls_dir}")
        return valid

    def load_split(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Load file paths and labels, then split into train/val/test.
        Returns a dict with keys 'train', 'val', 'test'.
        Each value is a list of (filepath, label_index) tuples.
        """
        if self.has_explicit_splits():
            splits = {
                split: self._load_samples_from_dir(self.local_path / split)
                for split in ("train", "val", "test")
            }
            if not any(splits.values()):
                raise ValueError(f"No samples found in explicit split directories under {self.local_path}")
            return splits

        all_samples = self._load_samples_from_dir(self.local_path)

        if not all_samples:
            raise ValueError(f"No samples found in {self.local_path}")

        paths = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]

        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths,
            labels,
            test_size=self.split_ratio["test"],
            random_state=self.seed,
            stratify=labels,
        )

        # Second split: train vs val
        val_ratio = self.split_ratio["val"] / (self.split_ratio["train"] + self.split_ratio["val"])
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=train_val_labels,
        )

        return {
            "train": list(zip(train_paths, train_labels)),
            "val": list(zip(val_paths, val_labels)),
            "test": list(zip(test_paths, test_labels)),
        }

    def dry_run(self) -> None:
        """Print class counts and sample shapes without loading images."""
        print(f"\n=== Dataset: {self.name} ===")
        valid = self.validate()
        if not valid:
            print("Validation FAILED. Dataset may not be downloaded.")
            return

        splits = self.load_split()
        for split_name, samples in splits.items():
            counts = {}
            for _, label in samples:
                counts[label] = counts.get(label, 0) + 1
            label_names = {v: k for k, v in self.classes.items()}
            print(f"  {split_name}: {len(samples)} samples")
            for label_idx, count in sorted(counts.items()):
                print(f"    - {label_names[label_idx]}: {count}")


def main():
    """Dry-run all manifests."""
    manifest_dir = Path(__file__).parent.parent / "manifests"
    for manifest_file in manifest_dir.glob("*.json"):
        manifest = DatasetManifest(str(manifest_file))
        manifest.dry_run()


if __name__ == "__main__":
    main()
