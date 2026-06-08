# OralPath Implementation Plan

we are building this for a non commercial purposes

## Purpose

This plan translates `VISION.md` and `VISION_DIRECTOR.md` into the order of work I would actually execute.

The current project is pre-development. The correct first move is not to build every app feature at once. The first move is to prove the model path, create a thin Android vertical slice around it, and keep every later feature attached to measurable evidence.

## Project Principle

OralPath v1 is not a generic Android app wrapped around a newly trained EfficientNet model.

OralPath v1 is a mobile-first OSCC diagnostic research tool built from:

- Frozen pathology foundation backbones: UNI or CTransPath
- OralPatho architecture and training scripts for Models 1 and 2
- Kaggle OSCC data for binary OSCC detection
- ORCHID data for grading and optional OSMF support
- A deferred MobileViT segmentation path for v1.2

The implementation should protect that shape from day one.

## Version Scope

### v0.1: Research Scaffold

Goal: Create the repo, scripts, documentation, model interfaces, and first runnable experiments.

Included:

- Python model workspace
- Dataset manifests
- OralPatho clone/adaptation notes
- Model contract JSON schema
- Android app skeleton
- Placeholder/mock inference
- Test and evaluation harness

Not included:

- Production clinical use
- Full segmentation
- Cloud sync
- iOS
- Regulatory submission

### v1: Detection And Grading Pilot

Goal: A doctor can capture or upload a microscope image, run OSCC detection and grading, view confidence and heatmap output, store the case locally, and export a basic report.

Included:

- Model 1: OSCC vs normal
- Model 2: grading, ideally normal / OSMF / WD / MD / PD if v1 decision allows OSMF
- On-device preprocessing
- API inference path
- Offline model fallback if model size and accuracy allow
- Local case storage
- PDF report

Deferred:

- Model 3 segmentation
- Whole-slide support
- Multi-image case aggregation
- iOS
- ABDM integration

### v1.2: Segmentation

Goal: Add tissue/TIL segmentation using TILSeg-MobileViT trained on NDB-UFES.

Included:

- MobileViT segmentation model
- TIL density or tissue-region overlay
- Segmentation metrics
- UI support for component breakdown

## Repository Structure To Create

```text
oralpath/
  android/
    app/
  model/
    data/
      manifests/
      preprocessing/
    training/
      stage1_detection/
      stage2_grading/
      stage3_segmentation/
    export/
    evaluation/
    notebooks/
    external/
      oralpatho/
      orchid/
  docs/
    VISION.md
    VISION_DIRECTOR.md
    IMPLEMENTATION_PLAN.md
    ARCHITECTURE.md
    MODEL_CARD.md
    DATASETS.md
    REGULATORY.md
    PATENT_STRATEGY.md
  scripts/
  tests/
```

If this repo remains rooted at the current folder, create these directories directly under the current root and keep the existing docs at the top level or move them into `docs/` after confirming that document paths should change.

## Workstream 1: Research And Evidence Setup

### Step 1: Confirm Source Assets

Create `docs/DATASETS.md` with evidence links, license notes, access instructions, and local download expectations for:

| Asset | Role | Required Evidence |
|---|---|---|
| OralPatho | Model 1 and 2 architecture/training reference | Repo URL, MIT license, scripts inventory |
| ORCHID | Model 2 training data | CC BY 4.0 license, class split, patch format |
| Kaggle OSCC | Model 1 training data | CC BY 4.0 license, normal/cancer counts |
| NDB-UFES | v1.2 segmentation data | License, annotations, patch format |
| UNI | Frozen feature extractor candidate | License, access method, embedding shape |
| CTransPath | Frozen feature extractor candidate | GPLv3-NC license, model loading path |
| Path Foundation | Benchmark candidate | Access constraints and license |

Output:

- `docs/DATASETS.md`
- `model/data/manifests/*.json`
- License inventory table

Acceptance gate:

- Every dataset/model used in training has a source URL, license, expected class labels, and local path convention.

### Step 2: Define The Model Contract

Before training, define what the Android app expects from inference.

Create `docs/ARCHITECTURE.md` and a JSON schema for inference output:

```json
{
  "case_id": "string",
  "model_version": "string",
  "input_quality": {
    "usable": true,
    "blur_score": 0.0,
    "coverage_score": 0.0
  },
  "stage1": {
    "label": "normal|oscc",
    "confidence": 0.0
  },
  "stage2": {
    "label": "normal|osmf|wd|md|pd|null",
    "confidence": 0.0
  },
  "heatmap": {
    "available": true,
    "uri": "string|null"
  },
  "disclaimer": "research_use_only"
}
```

Acceptance gate:

- Android, API inference, ONNX export, and TFLite fallback all use the same contract.

## Workstream 2: Model 1 Detection

### Step 3: Clone And Isolate OralPatho

Place OralPatho under `model/external/oralpatho/` or add it as a submodule if the repo should track upstream updates.

Do not rewrite OralPatho first. Inventory it.

Output:

- `docs/ORALPATHO_ADAPTATION.md`
- List of Stage 1 entry points
- List of assumptions made for WSI input
- Patch-input adaptation notes

Acceptance gate:

- We can identify exactly where OralPatho Stage 1 loads data, extracts features, trains heads, and evaluates results.

### Step 4: Build Kaggle OSCC Patch Loader

Create a dataset loader that maps Kaggle OSCC images into the Stage 1 expected format.

Required behavior:

- Normal/cancer label mapping
- Train/validation/test split
- Macenko normalization option
- Augmentation option
- Reproducible seed
- Manifest export

Acceptance gate:

- A dry-run command prints class counts and sample tensor shapes without starting training.

### Step 5: Frozen Backbone Stage 1 Experiment

Train only the classification head on top of frozen UNI or CTransPath features.

Primary run:

- Backbone: UNI or CTransPath
- Frozen: yes
- Dataset: Kaggle OSCC
- Target: sensitivity >= 0.95, specificity >= 0.90

Fallback run:

- Backbone: EfficientNetB3
- Purpose: benchmark and emergency fallback if OralPatho adaptation misses by more than 5%

Acceptance gate:

- Evaluation report includes AUC-ROC, sensitivity, specificity, F1, confusion matrix, and threshold used.

## Workstream 3: Model 2 Grading

### Step 6: ORCHID Training Integration

Adapt OralPatho Stage 2 using ORCHID training scripts.

Class strategy:

- Minimum v1: WD / MD / PD after OSCC detection
- Stronger v1: normal / OSMF / WD / MD / PD

The OSMF decision should be resolved early because ORCHID and OralPatho support it naturally. If included, the app can display OSMF as a pre-malignant finding with careful disclaimer language.

Acceptance gate:

- Class labels are fixed in one shared config used by training, evaluation, export, and Android UI.

### Step 7: Train Stage 2 Head

Train the grading classifier on ORCHID patches.

Required metrics:

- Overall accuracy
- Per-class precision/recall/F1
- Confusion matrix
- Weighted F1
- Specific attention to WD vs MD and MD vs PD confusion

Acceptance gate:

- Grading accuracy >= 0.85 or a written failure analysis explains the gap.

## Workstream 4: Export And Inference

### Step 8: Export Pipeline

Create repeatable exports for:

- ONNX for server-side inference
- TFLite INT8 for on-device fallback if supported by the selected backbone/head shape

If the foundation backbone is too large for mobile fallback, the offline path should use EfficientNetB3 or another compact benchmark model until the foundation path is compressed.

Acceptance gate:

- Same sample image produces consistent labels between PyTorch and exported runtime within acceptable numerical tolerance.

### Step 9: Minimal Inference Service

Implement a small FastAPI inference service before relying on hosted APIs.

Purpose:

- Lock the inference contract
- Test upload flow
- Generate heatmap artifact references
- Provide a local target for Android integration

Acceptance gate:

- `POST /predict` accepts an image and returns the model contract JSON.

## Workstream 5: Android Vertical Slice

### Step 10: Android Skeleton

Create the Android app with:

- Kotlin
- Jetpack Compose
- MVVM + Clean Architecture
- Hilt
- Navigation
- CameraX
- Room
- Retrofit/OkHttp

First screens:

- Capture/upload screen
- Result screen
- Case history screen
- Report preview screen

Acceptance gate:

- App runs with mock inference and stores a mock case locally.

### Step 11: Preprocessing

Implement on-device preprocessing:

- Blur score
- Coverage check
- Resize and normalization
- Macenko stain normalization if feasible on Android performance budget

Acceptance gate:

- Bad input images are flagged before inference.
- Preprocessing output shape matches model input shape.

### Step 12: Real Inference Integration

Wire Android to:

- Primary API inference path
- Offline fallback path if a model is available

Acceptance gate:

- A captured or uploaded image returns a real prediction and renders the result card.

## Workstream 6: Clinical Output

### Step 13: Result UX

Result screen must show:

- Classification
- Grade or OSMF label if enabled
- Confidence
- Heatmap overlay if available
- Plain-language clinical caution
- Research-use disclaimer

Acceptance gate:

- Result language never claims final diagnosis.
- Result always recommends qualified pathologist review.

### Step 14: Case Storage And PDF

Implement local case storage:

- Patient ID
- Date
- Slide site
- Magnification
- Result
- Confidence
- Thumbnail
- Optional heatmap

Implement PDF export:

- Doctor/institution/date
- Captured image
- Heatmap if available
- AI classification and confidence
- Disclaimer

Acceptance gate:

- Case survives app restart.
- PDF can be generated without network access.

## Workstream 7: Benchmarking And Validation

### Step 15: Benchmark Matrix

Benchmark:

| Candidate | Role |
|---|---|
| UNI | Primary foundation candidate |
| CTransPath | Primary foundation candidate |
| CONCH | Research comparison |
| Path Foundation | Research comparison |
| EfficientNetB3 | Fallback benchmark |
| Original OralPatho | Published architecture baseline |

Acceptance gate:

- Benchmark report explains which model is selected for v1 and why.

### Step 16: Pilot Readiness

Before any pilot:

- Ethics/disclaimer language reviewed
- Dataset and model licenses documented
- Validation protocol drafted
- Data export path anonymized
- No cloud sync of patient data by default

Acceptance gate:

- `docs/PILOT_READINESS.md` exists and lists open blockers.

## Suggested First Two Weeks

### Week 1

- Create repo structure
- Create `docs/DATASETS.md`
- Clone/inventory OralPatho
- Define inference JSON schema
- Build Kaggle OSCC dry-run loader
- Create Android mock app skeleton

### Week 2

- Run first Stage 1 frozen-backbone experiment
- Add EfficientNetB3 fallback benchmark
- Create local FastAPI mock/real inference endpoint
- Wire Android to mock API response
- Produce first evaluation report

## Decision Gates

| Gate | Decision |
|---|---|
| OSMF inclusion | Include in v1 if ORCHID labels are clean and app language can handle pre-malignant output safely |
| Backbone choice | Pick UNI or CTransPath based on accuracy, access, export feasibility, and runtime cost |
| Offline fallback | Use compact fallback if foundation model is too large for phone inference |
| Segmentation | Keep out of v1 unless detection/grading finish early and Model 3 metrics are reproducible |
| Pilot readiness | Do not pilot until model card, dataset licenses, disclaimer, and validation protocol are complete |

## Definition Of Done For v1

v1 is done when:

- Model 1 detects OSCC vs normal with target sensitivity/specificity on held-out data.
- Model 2 grades WD/MD/PD, and optionally OSMF, with documented per-class metrics.
- Android can capture/upload an image, run inference, show results, save the case, and export a report.
- The model card documents datasets, labels, metrics, limitations, and intended use.
- The app clearly states research-use-only and pathologist-review-required disclaimers.
- Benchmarking explains why the chosen backbone is better than the fallback.
- Model 3 segmentation is explicitly marked v1.2 unless separately completed and validated.

## Immediate Next Action

Start with the research scaffold and model contract.

The first implementation commit should create:

- `docs/DATASETS.md`
- `docs/ARCHITECTURE.md`
- `docs/MODEL_CARD.md`
- `model/data/manifests/`
- `model/training/stage1_detection/`
- `model/evaluation/`
- A minimal Android project or a placeholder `android/README.md` if Android scaffolding will be generated later

That gives the project a backbone before the codebase grows around assumptions.
