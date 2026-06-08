"""
OralPath — Stage 1 Training Script
Binary classification: OSCC vs Normal tissue.

Usage:
    python model/training/stage1_detection/train.py --config config.yaml

Backbone options (frozen):
    - uni:          HuggingFace mahmoodlab/UNI
    - ctranspath:   CTransPath (academic download)
    - efficientnetb3: timm efficientnet_b3 (fallback benchmark)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ModuleNotFoundError:
    torch = None
    optim = None
    DataLoader = None
    transforms = None
    nn = None
    Dataset = object

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.data.preprocessing.dataset_loader import DatasetManifest


class OSCCDetectionModel(nn.Module if nn is not None else object):
    """Stage 1: Binary OSCC detection with frozen backbone + trainable head."""

    def __init__(
        self,
        backbone_name: str = "uni",
        freeze_backbone: bool = True,
        head_hidden_dim: int = 512,
        dropout: float = 0.3,
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
            nn.Linear(head_hidden_dim, 1),
        )

    def _load_backbone(self, name: str):
        """Load backbone and return (model, feature_dim)."""
        if name == "efficientnetb3":
            try:
                import timm
                model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
                return model, 1536, "efficientnetb3"
            except ImportError:
                raise ImportError("timm is required for EfficientNetB3. Install: pip install timm")

        elif name == "uni":
            try:
                from transformers import AutoModel, AutoConfig
                model = AutoModel.from_pretrained("mahmoodlab/UNI", trust_remote_code=True)
                return model, 1024, "uni"
            except Exception as e:
                print(f"[WARN] Could not load UNI: {e}")
                print("[WARN] Falling back to EfficientNetB3")
                return self._load_backbone("efficientnetb3")

        elif name == "ctranspath":
            # CTransPath requires academic download; placeholder
            print("[WARN] CTransPath not yet loaded. Using EfficientNetB3 placeholder.")
            return self._load_backbone("efficientnetb3")

        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x):
        if self.actual_backbone_name == "uni":
            # UNI returns last_hidden_state
            features = self.backbone(x).last_hidden_state[:, 0, :]
        else:
            features = self.backbone(x)
        return self.head(features).squeeze(-1)


class ImageSplitDataset(Dataset):
    """Simple image dataset backed by DatasetManifest split samples."""

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
        return image, torch.tensor(label, dtype=torch.float32)


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or transforms is None:
        raise RuntimeError(
            "PyTorch/torchvision are required for training. Install dependencies with "
            "`pip install -r requirements.txt` or run this command from the Colab runtime."
        )


def get_transforms(config: dict, split: str = "train"):
    """Build torchvision transforms from config."""
    target_size = tuple(config["dataset"]["target_size"])
    aug = config.get("augmentation", {})

    if split == "train":
        tfm_list = [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip() if aug.get("horizontal_flip") else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug.get("rotation_degrees", 0)) if aug.get("rotation_degrees") else transforms.Lambda(lambda x: x),
        ]
        if aug.get("color_jitter"):
            cj = aug["color_jitter"]
            tfm_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0),
                    contrast=cj.get("contrast", 0),
                    saturation=cj.get("saturation", 0),
                    hue=cj.get("hue", 0),
                )
            )
        tfm_list.append(transforms.ToTensor())
    else:
        tfm_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]

    # Normalize for ImageNet / foundation model
    tfm_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return transforms.Compose(tfm_list)


def build_loaders(config: dict):
    require_torch()
    manifest = DatasetManifest(config["dataset"]["manifest"])
    splits = manifest.load_split()

    batch_size = int(config["dataset"]["batch_size"])
    num_workers = int(config["dataset"].get("num_workers", 0))

    train_dataset = ImageSplitDataset(splits["train"], get_transforms(config, "train"))
    val_dataset = ImageSplitDataset(splits["val"], get_transforms(config, "val"))
    test_dataset = ImageSplitDataset(splits["test"], get_transforms(config, "test"))

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def iter_limited(loader, max_batches: int | None):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def evaluate(model, loader, criterion, device, max_batches: int | None = None):
    model.eval()
    losses = []
    labels = []
    probs = []

    with torch.no_grad():
        for images, targets in iter_limited(loader, max_batches):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))
            labels.extend(targets.detach().cpu().numpy().astype(int).tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

    predictions = [1 if value >= 0.5 else 0 for value in probs]
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(labels, predictions)) if labels else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)) if labels else 0.0,
        "recall": float(recall_score(labels, predictions, zero_division=0)) if labels else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)) if labels else 0.0,
        "auc_roc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.0,
    }

    if len(set(labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
        metrics["confusion_matrix"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    else:
        metrics["sensitivity"] = 0.0
        metrics["specificity"] = 0.0
        metrics["confusion_matrix"] = []

    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches: int | None = None):
    model.train()
    losses = []

    for images, targets in iter_limited(loader, max_batches):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train OralPath Stage 1 detection model")
    parser.add_argument("--config", type=str, default="model/training/stage1_detection/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per split for smoke tests")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Backbone: {config['model']['backbone']} (frozen={config['model']['freeze_backbone']})")
    print(f"Dataset: {config['dataset']['manifest']}")
    print("=" * 50)

    if args.dry_run:
        print("\n[Dry run] Config loaded successfully.")
        print(json.dumps(config, indent=2))
        manifest = DatasetManifest(config["dataset"]["manifest"])
        manifest.dry_run()
        return

    require_torch()
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(config)
    model = OSCCDetectionModel(
        backbone_name=config["model"]["backbone"],
        freeze_backbone=bool(config["model"]["freeze_backbone"]),
        head_hidden_dim=int(config["model"]["head_hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
    )

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    log_dir = Path(config["logging"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_path = checkpoint_dir / "stage1_best.pt"
    history = []
    patience = int(config["training"]["early_stopping_patience"])
    stale_epochs = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, args.max_batches)
        val_metrics = evaluate(model, loaders["val"], criterion, device, args.max_batches)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "backbone_requested": config["model"]["backbone"],
            "backbone_actual": model.actual_backbone_name,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['auc_roc']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f}"
        )

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "backbone_actual": model.actual_backbone_name,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    test_metrics = evaluate(model, loaders["test"], criterion, device, args.max_batches)
    report = {
        "experiment_name": config["experiment_name"],
        "best_val_auc": best_auc,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(best_path),
    }

    report_path = log_dir / "stage1_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Report: {report_path}")
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
