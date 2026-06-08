"""
Local dry-run: validate training pipeline without full model/data.
Uses EfficientNetB3 as lightweight backbone stand-in for UNI.
Tests: data loading, transforms, forward, backward, shapes.
"""
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from PIL import Image

IMG_SIZE = 224
BATCH_SIZE = 8
DEVICE = torch.device("cpu")
DATA_ROOT = Path("model/data/raw/kaggle_oscc")


class KaggleOSCCDataset(Dataset):
    """Minimal dataset for dry-run."""
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = ["Normal", "OSCC"]
        self.class_to_idx = {"Normal": 0, "OSCC": 1}
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if cls_dir.exists():
                for img_path in cls_dir.iterdir():
                    if img_path.is_file():
                        self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    print("=" * 50)
    print("ORALPATH PIPELINE DRY-RUN")
    print("=" * 50)

    # 1. Data loading
    print("\n[1/5] Loading Kaggle OSCC (tiny subset)...")
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = KaggleOSCCDataset(DATA_ROOT, split="train", transform=tfm)
    print(f"  Total train images: {len(ds)}")
    tiny_ds = Subset(ds, range(min(32, len(ds))))
    loader = DataLoader(tiny_ds, batch_size=BATCH_SIZE, shuffle=True)
    images, labels = next(iter(loader))
    print(f"  Batch shape: {images.shape}, labels: {labels.shape}")
    print(f"  Classes in batch: {labels.unique().tolist()}")

    # 2. Model build (EfficientNetB3 as UNI stand-in)
    print("\n[2/5] Building model (EfficientNetB3 stand-in)...")
    try:
        import timm
        backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        feat_dim = 1536
    except ImportError:
        print("  timm not installed, using dummy backbone")
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * IMG_SIZE * IMG_SIZE, 512))
        feat_dim = 512

    for p in backbone.parameters():
        p.requires_grad = False

    model = nn.Sequential(
        backbone,
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total")

    # 3. Forward pass
    print("\n[3/5] Forward pass...")
    model.train()
    outputs = model(images.to(DEVICE))
    print(f"  Output shape: {outputs.shape}, range: [{outputs.min():.3f}, {outputs.max():.3f}]")

    # 4. Loss + backward
    print("\n[4/5] Loss computation + backward...")
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, labels.float().to(DEVICE))
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")

    has_grad = [name for name, p in model.named_parameters() if p.grad is not None]
    print(f"  Layers with grad: {len(has_grad)}")

    # 5. Optimizer step
    print("\n[5/5] Optimizer step...")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt.step()
    print("  Step OK")

    print("\n" + "=" * 50)
    print("[PASS] All pipeline checks passed!")
    print("Ready for Colab training with UNI backbone.")
    print("=" * 50)


if __name__ == "__main__":
    main()
