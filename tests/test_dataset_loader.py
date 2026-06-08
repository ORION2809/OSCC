import json
from pathlib import Path

from PIL import Image

from model.data.preprocessing.dataset_loader import DatasetManifest


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def test_manifest_loads_explicit_splits_with_case_insensitive_classes(tmp_path):
    dataset_root = tmp_path / "kaggle_oscc"
    for split in ("train", "val", "test"):
        _write_image(dataset_root / split / "Normal" / f"{split}_normal.jpg")
        _write_image(dataset_root / split / "OSCC" / f"{split}_oscc.jpg")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "Kaggle OSCC",
                "local_path": str(dataset_root),
                "classes": {"normal": 0, "oscc": 1},
                "split": {"train": 0.7, "val": 0.15, "test": 0.15},
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    manifest = DatasetManifest(str(manifest_path))

    assert manifest.validate()
    splits = manifest.load_split()
    assert {name: len(samples) for name, samples in splits.items()} == {
        "train": 2,
        "val": 2,
        "test": 2,
    }
    assert sorted(label for _, label in splits["train"]) == [0, 1]


def test_manifest_loads_class_aliases_for_orchid_style_names(tmp_path):
    dataset_root = tmp_path / "orchid"
    for split in ("train", "val", "test"):
        _write_image(dataset_root / split / "normal" / f"{split}_normal.png")
        _write_image(dataset_root / split / "osmf" / f"{split}_osmf.png")
        _write_image(dataset_root / split / "wdoscc" / f"{split}_wd.png")
        _write_image(dataset_root / split / "mdoscc" / f"{split}_md.png")
        _write_image(dataset_root / split / "pdoscc" / f"{split}_pd.png")

    manifest_path = tmp_path / "orchid_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "ORCHID",
                "local_path": str(dataset_root),
                "classes": {"normal": 0, "osmf": 1, "wd": 2, "md": 3, "pd": 4},
                "class_aliases": {"wd": ["wdoscc"], "md": ["mdoscc"], "pd": ["pdoscc"]},
                "split": {"train": 0.7, "val": 0.15, "test": 0.15},
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    manifest = DatasetManifest(str(manifest_path))

    assert manifest.validate()
    splits = manifest.load_split()
    assert len(splits["train"]) == 5
    assert sorted(label for _, label in splits["train"]) == [0, 1, 2, 3, 4]
