"""OralPath Level 1 inference wrapper.

Loads a Stage 2 checkpoint and returns the five-class classification
contract for a single input image.

Usage:
    python model/inference/stage2_predict.py \
        --image path/to/image.png \
        --checkpoint model/training/stage2_grading/checkpoints/stage2_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision import transforms

from model.training.stage2_grading.train import Stage2GradingModel


# Map canonical checkpoint class names to user-facing Level 1 labels.
CLASS_NAME_MAP = {
    "normal": "normal",
    "osmf": "osmf",
    "wd": "wdoscc",
    "wdoscc": "wdoscc",
    "md": "mdoscc",
    "mdoscc": "mdoscc",
    "pd": "pdoscc",
    "pdoscc": "pdoscc",
}


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    class_names = checkpoint.get("class_names") or config.get("evaluation", {}).get(
        "class_names", ["normal", "osmf", "wd", "md", "pd"]
    )

    model = Stage2GradingModel(
        backbone_name=model_cfg.get("backbone", "uni"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        num_classes=int(model_cfg.get("num_classes", len(class_names))),
    ).to(device)

    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "head_state_dict"
    if state_key == "head_state_dict":
        model.head.load_state_dict(checkpoint["head_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model, class_names, config


def build_transform(target_size: tuple[int, int] = (224, 224)):
    return transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(
    image_path: str,
    checkpoint_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, config = load_model(checkpoint_path, device)

    image = Image.open(image_path).convert("RGB")
    transform = build_transform(target_size)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    user_label = CLASS_NAME_MAP.get(predicted_class, predicted_class)

    class_probabilities = {
        CLASS_NAME_MAP.get(name, name): float(probabilities[idx])
        for idx, name in enumerate(class_names)
    }

    # Ensure all five Level 1 keys exist even if checkpoint uses short names.
    for key in ("normal", "osmf", "wdoscc", "mdoscc", "pdoscc"):
        class_probabilities.setdefault(key, 0.0)

    return {
        "label": user_label,
        "confidence": float(probabilities[predicted_index]),
        "class_probabilities": class_probabilities,
        "model_version": config.get("experiment_name", "stage2_grading_v1"),
        "disclaimer": "research_use_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OralPath Level 1 inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stage 2 checkpoint (best or last)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Model input size (H W)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to write result to",
    )
    args = parser.parse_args()

    result = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        target_size=tuple(args.target_size),
    )

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Result written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
