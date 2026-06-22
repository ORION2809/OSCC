"""
OralPath — Dataset Loader
Shared dataset loading utilities for Kaggle OSCC and ORCHID datasets.

Supports both directory-based datasets and zip-backed datasets. For zip-backed
manifests, sample paths are encoded as ``zip://path/to.zip!entry/name.png`` and
training code uses :func:`open_image_from_zip_path` to read them.
"""

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
ZIP_PREFIX = "zip://"


def encode_zip_path(zip_path: Path | str, entry_name: str) -> str:
    """Return a portable string reference to a file inside a zip archive."""
    return f"{ZIP_PREFIX}{zip_path}!{entry_name}"


def decode_zip_path(path: str) -> Tuple[Path, str]:
    """Parse a ``zip://...!entry`` path back into (zip_path, entry_name)."""
    if not path.startswith(ZIP_PREFIX):
        raise ValueError(f"Not a zip path: {path}")
    rest = path[len(ZIP_PREFIX):]
    if "!" not in rest:
        raise ValueError(f"Zip path missing entry separator '!': {path}")
    zip_str, entry_name = rest.split("!", 1)
    return Path(zip_str), entry_name


def open_image_from_zip_path(path: str) -> Image.Image:
    """Open an image encoded as ``zip://zip_path!entry_name``."""
    zip_path, entry_name = decode_zip_path(path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(entry_name) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")


class DatasetManifest:
    """Loads and validates a dataset manifest JSON."""

    def __init__(self, manifest_path: str):
        with open(manifest_path, "r") as f:
            self.config = json.load(f)
        self.name = self.config["name"]
        override_env = self.config.get("local_path_env")
        override_path = os.environ.get(override_env) if override_env else None
        self.local_path = Path(override_path or self.config["local_path"])
        self.classes = self.config["classes"]
        self.class_aliases = self.config.get("class_aliases", {})
        self.explicit_split_dirs = self.config.get("explicit_split_dirs", {})
        self.split_ratio = self.config["split"]
        self.seed = self.config["seed"]
        self.zip_splits = self.config.get("zip_splits")

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
                for fp in cls_dir.rglob(ext):
                    samples.append((str(fp), label_idx))
        return samples

    def _match_class_from_zip_entry(self, entry_name: str) -> str | None:
        """Return canonical class name for a zip entry based on its path parts."""
        parts = [p.lower() for p in Path(entry_name).parts]
        for cls_name in self.classes:
            if cls_name.lower() in parts:
                return cls_name
            for alias in self.class_aliases.get(cls_name, []):
                if alias.lower() in parts:
                    return cls_name
        return None

    def _load_samples_from_zip(self, zip_path: Path) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        if not zip_path.exists():
            print(f"[ERROR] Zip file does not exist: {zip_path}")
            return samples
        with zipfile.ZipFile(zip_path, "r") as zf:
            for entry in zf.namelist():
                if not entry.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    continue
                cls_name = self._match_class_from_zip_entry(entry)
                if cls_name is None:
                    continue
                samples.append((encode_zip_path(zip_path, entry), self.classes[cls_name]))
        return samples

    def has_explicit_splits(self) -> bool:
        """Return true when dataset is already organized as train/val/test."""
        if self.zip_splits:
            return all(
                (self.local_path / self.zip_splits.get(split)).exists()
                for split in ("train", "val", "test")
                if self.zip_splits.get(split)
            )
        return all(self._resolve_split_dir(split) is not None for split in ("train", "val", "test"))

    def _resolve_split_dir(self, split: str) -> Path | None:
        configured = self.explicit_split_dirs.get(split)
        if configured:
            path = self.local_path / configured
            if path.exists():
                return path

        candidates = [
            self.local_path / split,
            self.local_path / f"ORCHID_{split}" / split,
        ]
        if split == "val":
            candidates.extend(
                [
                    self.local_path / "validation",
                    self.local_path / "ORCHID_val" / "val",
                    self.local_path / "ORCHID_validation" / "validation",
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_zip_split(self, split: str) -> Path | None:
        if not self.zip_splits:
            return None
        zip_name = self.zip_splits.get(split)
        if not zip_name:
            return None
        zip_path = self.local_path / zip_name
        return zip_path if zip_path.exists() else None

    def validate(self) -> bool:
        """Check that expected directories or zip files exist."""
        if not self.local_path.exists():
            print(f"[ERROR] Dataset path does not exist: {self.local_path}")
            return False

        if self.zip_splits:
            valid = True
            for split in ("train", "val", "test"):
                zip_path = self._resolve_zip_split(split)
                if zip_path is None:
                    print(f"[ERROR] Zip for {split} not found under {self.local_path}")
                    valid = False
                else:
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            namelist = zf.namelist()
                            print(f"[OK] {split} zip: {zip_path} ({len(namelist)} entries)")
                    except zipfile.BadZipFile:
                        print(f"[ERROR] Corrupt zip: {zip_path}")
                        valid = False
            return valid

        bases = [self.local_path]
        if self.has_explicit_splits():
            bases = [self._resolve_split_dir(split) for split in ("train", "val", "test")]

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
        if self.zip_splits:
            splits = {}
            for split in ("train", "val", "test"):
                zip_path = self._resolve_zip_split(split)
                if zip_path is None:
                    raise ValueError(f"Zip for {split} not found under {self.local_path}")
                splits[split] = self._load_samples_from_zip(zip_path)
            if not any(splits.values()):
                raise ValueError(f"No samples found in zip files under {self.local_path}")
            return splits

        if self.has_explicit_splits():
            splits = {
                split: self._load_samples_from_dir(self._resolve_split_dir(split))
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
