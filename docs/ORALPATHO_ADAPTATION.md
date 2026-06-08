# OralPatho — Adaptation Notes for OralPath

> Inventory and adaptation plan for integrating OralPatho into OralPath's mobile-first pipeline.

## Source Repository

| Attribute | Value |
|---|---|
| **Repository** | https://github.com/NishaChaudhary23/oralpatho |
| **License** | MIT |
| **Published** | 2025, medRxiv |
| **Local path** | `model/external/oralpatho/` |

---

## What OralPatho Does

OralPatho is a **whole-slide image (WSI) MIL framework**:

1. Extracts 256×256 patches from WSIs
2. Encodes patches into 512-dim vectors using a modified ResNet50
3. Aggregates patch embeddings with attention-based MIL
4. Produces slide-level predictions (binary + multiclass grading)

**Published performance:**
- Binary F1 > 0.93
- Multiclass macro-F1: 0.72 (WD), 0.70 (MD), 0.68 (PD)
- External validation AUCs: 0.871 (WD), 0.823 (MD), 0.731 (PD)

---

## Key Files Inventory

### Models

| File | Purpose | Relevance to OralPath |
|---|---|---|
| `src/models/mil_models.py` | Attention MIL, GatedAttention MIL, SelfAttention | **Reference only** — MIL assumes multiple patches per slide; OralPath v1 processes single patches |
| `src/models/mil_models_multiclass.py` | Multiclass gated attention with top-K selection | **Reference only** — same MIL architecture |

### Training

| File | Purpose | Relevance to OralPath |
|---|---|---|
| `src/training/binary/train_binary.py` | Binary MIL training loop | **Adapt** — loss function, metrics, scheduler patterns are reusable |
| `src/training/multiclass/train_multiclass.py` | Multiclass MIL training loop | **Adapt** — same as above |

### Datasets

| File | Purpose | Relevance to OralPath |
|---|---|---|
| `src/datasets/wsi_datasets.py` | HDF5 feature bag loader for binary | **Replace** — OralPath loads raw images, not pre-extracted features |
| `src/datasets/wsi_datasets_multiclass.py` | HDF5 feature bag loader for multiclass | **Replace** — same |

### Preprocessing

| File | Purpose | Relevance to OralPath |
|---|---|---|
| `src/preprocessing/extract_patches.ipynb` | WSI patch extraction | **Reference** — patch size and overlap strategy |
| `src/preprocessing/Feature_extraction_resnet50.ipynb` | ResNet50 feature extraction | **Replace** — OralPath uses frozen foundation backbones (UNI/CTransPath) |
| `src/preprocessing/normalization.ipynb` | Stain normalization | **Adapt** — Macenko or similar method needed on Android |

### Testing

| File | Purpose | Relevance to OralPath |
|---|---|---|
| `testing/test_binary.py` | Binary inference on WSIs | **Adapt** — evaluation metrics are reusable |
| `testing/test_multiclass_cv.py` | Multiclass inference | **Adapt** — same |

---

## Adaptation Strategy

### Core Difference: WSI MIL → Single-Patch Classification

OralPatho operates on **bags of patches** (MIL). A WSI produces hundreds of patches; the model attends over all patches to make a single slide-level prediction.

OralPath v1 operates on **single images** from a phone camera. The user captures one field of view through the microscope. There is no WSI and no bag of patches.

**Therefore:** OralPath cannot use MIL directly. It must adapt OralPatho's architectural principles (attention, gating, top-K selection) into a standard single-image classifier.

### Proposed Adaptation

| OralPatho Component | OralPath Adaptation |
|---|---|
| ResNet50 patch encoder (512-dim) | Replace with frozen UNI (1024-dim) or CTransPath |
| Attention MIL aggregator | Remove — single image means single feature vector |
| Classification head (binary) | Train new linear head on top of frozen backbone features |
| Gated attention (multiclass) | Remove MIL; train new multi-class head on frozen features |
| Top-K patch selection | Not applicable for single-patch input |
| BCEWithLogitsLoss + pos_weight | **Reusable** — same loss for binary stage |
| CrossEntropyLoss (multiclass) | **Reusable** — same loss for grading stage |
| CosineAnnealingLR | **Reusable** — same scheduler |
| Metrics (F1, kappa, MCC, confusion matrix) | **Reusable** — same evaluation harness |

### Architecture Comparison

```
OralPatho (WSI-level):
  WSI → [extract N patches] → [ResNet50 encoder] → N×512 embeddings
  → [Attention MIL] → aggregated representation → classifier → slide label

OralPath v1 (patch-level):
  Single image → [frozen UNI/CTransPath] → 1×1024 embedding
  → [trainable linear head] → label
```

### What We Keep from OralPatho

1. **Loss functions** — BCEWithLogitsLoss (binary) with class weighting; CrossEntropyLoss (multiclass)
2. **Optimizer settings** — AdamW, LR 1e-4, weight decay ~1e-3
3. **Scheduler** — CosineAnnealingLR
4. **Metrics** — Accuracy, precision, recall, F1, Cohen's kappa, MCC, confusion matrix
5. **3-fold stratified cross-validation** — same validation strategy
6. **Weight initialization** — Kaiming uniform for new heads

### What We Change

1. **Input** — Single 224×224 or 300×300 image instead of WSI patch bags
2. **Encoder** — UNI or CTransPath instead of ResNet50
3. **Aggregator** — Removed; single image = single embedding
4. **Head** — Simpler: Linear → ReLU → Dropout → Linear (no MIL attention)
5. **Batch size** — Can use standard batching (e.g., 32) instead of batch_size=1 per slide

---

## ORCHID Integration

The ORCHID repository (`model/external/orchid/`) contains:

| File | Purpose |
|---|---|
| `training/train-classify-normal-osmf-oscc.py` | 3-class trainer (normal / OSMF / OSCC) |
| `training/train-classify-wdoscc-mdoscc-pdoscc.py` | 3-class trainer (WD / MD / PD) |
| `tools/patch-generator.py` | Patch generation from WSIs |
| `tools/color_normalisation.py` | Color normalization |
| `tools/split.py` | Train/val/test splitting |

**Adaptation:** ORCHID scripts use standard CNN classifiers (not MIL). They are closer to what OralPath needs. The main change is replacing their CNN backbone with a frozen foundation model.

---

## Implementation Plan

### Step 1: Stage 1 Binary (OSCC vs Normal)

1. Create `ImageDataset` that loads Kaggle OSCC images directly
2. Use frozen UNI/CTransPath as feature extractor
3. Add small trainable head (Linear → ReLU → Dropout → Linear → Sigmoid)
4. Train with BCEWithLogitsLoss + class weights
5. Evaluate with OralPatho's metric set

### Step 2: Stage 2 Grading (Normal / OSMF / WD / MD / PD)

1. Create `ImageDataset` that loads ORCHID images
2. Use same frozen backbone as Stage 1
3. Add multi-class head (Linear → ReLU → Dropout → Linear → Softmax)
4. Train with CrossEntropyLoss + label smoothing
5. Evaluate with per-class metrics

### Step 3: Export

1. Export backbone + head to ONNX
2. Attempt TFLite INT8 conversion for on-device fallback
3. If foundation model is too large, fall back to EfficientNetB3 for mobile

---

## Risk: Accuracy Gap

| Risk | Mitigation |
|---|---|
| Single-patch accuracy < WSI-MIL accuracy | Single-patch is inherently harder (less context). Target is mobile usability, not WSI-level accuracy. Accept 5–10% gap if sensitivity ≥ 0.95 is met. |
| Foundation model too large for mobile | Use API path as primary; TFLite/EfficientNetB3 as offline fallback. |
| ORCHID patch format mismatch | Verify ORCHID patch size and magnification against Kaggle OSCC. Normalize both to same input size. |

---

## Bottom Line

OralPatho provides the **training protocol, loss functions, metrics, and validation strategy**. Its MIL architecture is **not directly usable** for single-image mobile inference. OralPath replaces the MIL encoder-aggregator with a frozen foundation backbone + simple trainable head, while preserving OralPatho's proven training discipline.

---

*Last updated: June 2026*
