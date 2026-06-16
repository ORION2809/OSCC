"""
OralPath -- Stage 2 Grading Script
Five-class classification: normal / OSMF / WD / MD / PD.

Usage:
    python model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml

Backbone options (frozen):
    - uni:          HuggingFace mahmoodlab/UNI
    - ctranspath:   CTransPath (academic download)
    - efficientnetb3: timm efficientnet_b3 (fallback benchmark)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms
except ModuleNotFoundError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    transforms = None
    Dataset = object

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.data.preprocessing.dataset_loader import DatasetManifest


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or transforms is None:
        raise RuntimeError(
            "PyTorch/torchvision are required for training. Install dependencies with "
            "`pip install -r requirements.txt` or run this command from the Colab runtime."
        )


class Stage2GradingModel(nn.Module if nn is not None else object):
    """Stage 2: Five-class grading with frozen backbone + trainable head."""

    def __init__(
        self,
        backbone_name: str = "uni",
        freeze_backbone: bool = True,
        head_hidden_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 5,
    ):
        super().__init__()
        require_torch()
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim, self.actual_backbone_name = self._load_backbone(backbone_name)

        if freeze_backbone and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def _load_backbone(self, name: str):
        if name == "efficientnetb3":
            try:
                import timm

                model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
                return model, 1536, "efficientnetb3"
            except ImportError:
                raise ImportError("timm is required for EfficientNetB3. Install: pip install timm")

        if name == "uni":
            try:
                from transformers import AutoModel

                model = AutoModel.from_pretrained("mahmoodlab/UNI", trust_remote_code=True)
                return model, 1024, "uni"
            except Exception as exc:
                print(f"[WARN] Could not load UNI: {exc}")
                print("[WARN] Falling back to EfficientNetB3")
                return self._load_backbone("efficientnetb3")

        if name == "ctranspath":
            print("[WARN] CTransPath not yet loaded. Using EfficientNetB3 placeholder.")
            return self._load_backbone("efficientnetb3")

        raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x):
        if self.actual_backbone_name == "uni":
            features = self.backbone(x).last_hidden_state[:, 0, :]
        else:
            features = self.backbone(x)
        return self.head(features)


class ImageSplitDataset(Dataset):
    """Simple multiclass image dataset backed by DatasetManifest split samples."""

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(config: dict, split: str = "train"):
    require_torch()
    target_size = tuple(config["dataset"]["target_size"])
    aug = config.get("augmentation", {})

    if split == "train":
        transform_steps = [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip() if aug.get("horizontal_flip") else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug.get("rotation_degrees", 0))
            if aug.get("rotation_degrees")
            else transforms.Lambda(lambda x: x),
        ]
        if aug.get("color_jitter"):
            jitter = aug["color_jitter"]
            transform_steps.append(
                transforms.ColorJitter(
                    brightness=jitter.get("brightness", 0),
                    contrast=jitter.get("contrast", 0),
                    saturation=jitter.get("saturation", 0),
                    hue=jitter.get("hue", 0),
                )
            )
        transform_steps.append(transforms.ToTensor())
    else:
        transform_steps = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]

    transform_steps.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(transform_steps)


def compute_class_distribution(splits: dict[str, list], class_names: list[str]) -> dict:
    """Return sample counts per split and class, plus max/min train ratio."""
    distribution = {}
    for split_name, samples in splits.items():
        counts = {name: 0 for name in class_names}
        for _, label_idx in samples:
            counts[class_names[label_idx]] += 1
        distribution[split_name] = counts

    train_counts = list(distribution["train"].values())
    max_count = max(train_counts)
    min_count = min(train_counts)
    distribution["train_max_min_ratio"] = round(max_count / min_count, 3) if min_count > 0 else float("inf")
    return distribution


def build_class_weights(train_counts: dict[str, int], method: str = "inverse_frequency") -> torch.Tensor:
    """Compute per-class weights for CrossEntropyLoss."""
    counts = torch.tensor([train_counts[name] for name in train_counts], dtype=torch.float32)
    if method == "effective_samples":
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return weights


def build_loaders(config: dict, sampler: WeightedRandomSampler | None = None):
    require_torch()
    manifest = DatasetManifest(config["dataset"]["manifest"])
    splits = manifest.load_split()
    batch_size = int(config["dataset"]["batch_size"])
    num_workers = int(config["dataset"].get("num_workers", 0))

    return {
        "splits": splits,
        "train": DataLoader(
            ImageSplitDataset(splits["train"], get_transforms(config, "train")),
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            ImageSplitDataset(splits["val"], get_transforms(config, "val")),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            ImageSplitDataset(splits["test"], get_transforms(config, "test")),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def iter_limited(loader, max_batches: int | None):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches: int | None = None,
    epoch: int | None = None,
):
    model.train()
    losses = []

    for batch_index, (images, targets) in enumerate(iter_limited(loader, max_batches), start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if batch_index == 1 or batch_index % 25 == 0:
            prefix = f"Epoch {epoch:03d}" if epoch is not None else "Train"
            print(
                f"{prefix} batch {batch_index}/{len(loader)} "
                f"loss={float(loss.item()):.4f}",
                flush=True,
            )

    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, criterion, device, class_names: list[str], max_batches: int | None = None):
    model.eval()
    losses = []
    labels = []
    predictions = []

    with torch.no_grad():
        for images, targets in iter_limited(loader, max_batches):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))
            predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist())
            labels.extend(targets.detach().cpu().numpy().astype(int).tolist())

    present_labels = list(range(len(class_names)))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(labels, predictions)) if labels else 0.0,
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)) if labels else 0.0,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=present_labels).astype(int).tolist()
        if labels
        else [],
        "classification_report": classification_report(
            labels,
            predictions,
            labels=present_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        if labels
        else {},
    }


def main():
    parser = argparse.ArgumentParser(description="Train OralPath Stage 2 grading model")
    parser.add_argument("--config", type=str, default="model/training/stage2_grading/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print config and dataset counts, then exit")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per split for smoke tests")
    parser.add_argument("--max-epochs", type=int, default=None, help="Limit epochs for this run")
    parser.add_argument("--resume-state", type=str, default=None, help="Resume from a head-only training state")
    parser.add_argument("--state-output", type=str, default=None, help="Write head-only training state after each epoch")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    class_names = config["evaluation"]["class_names"]
    print("=" * 50)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Backbone: {config['model']['backbone']} (frozen={config['model']['freeze_backbone']})")
    print(f"Classes: {class_names}")
    print(f"Dataset: {config['dataset']['manifest']}")
    print("=" * 50)

    require_torch()
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset distribution preflight (Task 4.0) ---
    manifest = DatasetManifest(config["dataset"]["manifest"])
    splits = manifest.load_split()
    class_distribution = compute_class_distribution(splits, class_names)

    imbalance_cfg = config.get("imbalance", {})
    imbalance_mode = imbalance_cfg.get("mode", "none")
    auto_threshold = float(imbalance_cfg.get("auto_enable_threshold", 2.0))
    max_min_ratio = class_distribution["train_max_min_ratio"]

    if imbalance_mode == "auto":
        imbalance_mode = "weighted_loss" if max_min_ratio > auto_threshold else "none"

    print("\n[DATASET DISTRIBUTION]")
    for split_name in ("train", "val", "test"):
        counts = class_distribution[split_name]
        line = "  " + split_name + ": " + " / ".join(f"{cls}={counts[cls]}" for cls in class_names)
        print(line)
    print(f"  train max/min ratio = {max_min_ratio:.3f}")
    print(f"  imbalance handling = {imbalance_mode}")

    if args.dry_run:
        print("\n[Dry run] Config loaded successfully.")
        print(json.dumps(config, indent=2))
        manifest.dry_run()
        return

    sampler = None
    train_labels = [label_idx for _, label_idx in splits["train"]]
    if imbalance_mode == "weighted_sampler":
        class_counts = [train_labels.count(idx) for idx in range(len(class_names))]
        sample_weights = [1.0 / class_counts[idx] for idx in train_labels]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(train_labels),
            replacement=True,
        )
        print("[IMBALANCE] Using WeightedRandomSampler.", flush=True)

    loaders = build_loaders(config, sampler=sampler)

    model = Stage2GradingModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=bool(config["model"]["freeze_backbone"]),
        head_hidden_dim=int(config["model"]["head_hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        num_classes=int(config["model"]["num_classes"]),
    ).to(device)

    class_weights = None
    if imbalance_mode == "weighted_loss":
        class_weights = build_class_weights(class_distribution["train"], method=imbalance_cfg.get("weight_method", "inverse_frequency"))
        class_weights = class_weights.to(device)
        print(f"[IMBALANCE] Using weighted CrossEntropyLoss. weights={class_weights.cpu().tolist()}", flush=True)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
    )
    optimizer = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    log_dir = Path(config["logging"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    best_path = checkpoint_dir / "stage2_best.pt"
    state_path = Path(args.state_output) if args.state_output else checkpoint_dir / "stage2_last.pt"
    history = []
    stale_epochs = 0
    patience = int(config["training"]["early_stopping_patience"])
    start_epoch = 1

    if args.resume_state:
        resume_path = Path(args.resume_state)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device)
            model.head.load_state_dict(state["head_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            best_metric = float(state.get("best_metric", best_metric))
            history = list(state.get("history", []))
            stale_epochs = int(state.get("stale_epochs", 0))
            start_epoch = int(state["epoch"]) + 1
            print(f"[RESUME] Loaded {resume_path}; next epoch: {start_epoch}", flush=True)
        else:
            print(f"[RESUME] State file not found, starting fresh: {resume_path}", flush=True)

    configured_epochs = int(config["training"]["epochs"])
    epoch_count = configured_epochs
    if args.max_epochs is not None:
        epoch_count = min(epoch_count, start_epoch + args.max_epochs - 1)

    for epoch in range(start_epoch, epoch_count + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            args.max_batches,
            epoch,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device, class_names, args.max_batches)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_metrics,
                "backbone_requested": config["model"]["backbone"],
                "backbone_actual": model.actual_backbone_name,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}",
            flush=True,
        )

        current_metric = float(val_metrics.get("macro_f1", val_metrics["accuracy"]))
        if current_metric > best_metric:
            best_metric = current_metric
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "backbone_actual": model.actual_backbone_name,
                    "class_names": class_names,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        torch.save(
            {
                "head_state_dict": model.head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_metric": best_metric,
                "history": history,
                "stale_epochs": stale_epochs,
                "backbone_requested": config["model"]["backbone"],
                "backbone_actual": model.actual_backbone_name,
                "class_names": class_names,
            },
            state_path,
        )
        print(f"[STATE] Saved resumable state: {state_path}", flush=True)

        if stale_epochs >= patience:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break

    test_metrics = evaluate(model, loaders["test"], criterion, device, class_names, args.max_batches)
    report = {
        "experiment_name": config["experiment_name"],
        "best_val_metric": best_metric,
        "best_val_metric_name": "macro_f1",
        "class_distribution": class_distribution,
        "imbalance_handling": {
            "mode": imbalance_mode,
            "auto_threshold": auto_threshold,
            "train_max_min_ratio": max_min_ratio,
            "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
            "sampler": "WeightedRandomSampler" if sampler is not None else None,
        },
        "test": test_metrics,
        "history": history,
        "checkpoint": str(best_path),
        "class_names": class_names,
    }
    report_path = log_dir / "stage2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Report: {report_path}")
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
