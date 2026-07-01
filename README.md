<div align="center">
  <h1>🩺 OralPath — OSCC Diagnostic Assistant</h1>
  <p><strong>AI-powered histopathological analysis of Oral Squamous Cell Carcinoma from H&E-stained biopsy slides</strong></p>

  <p>
    <a href="https://github.com/ORION2809/OSCC/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT%20%2B%20Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://kotlinlang.org/"><img src="https://img.shields.io/badge/kotlin-2.0.21-7F52B0.svg?logo=kotlin&logoColor=white" alt="Kotlin"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://developer.android.com/jetpack/compose"><img src="https://img.shields.io/badge/jetpack%20compose-latest-4285F4.svg?logo=jetpackcompose&logoColor=white" alt="Jetpack Compose"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"></a>
    <a href="https://colab.research.google.com/"><img src="https://img.shields.io/badge/Colab-Training-F9AB00.svg?logo=googlecolab&logoColor=white" alt="Colab"></a>
    <a href="https://lightning.ai/"><img src="https://img.shields.io/badge/Lightning%20AI-792EE5.svg?logo=lightning&logoColor=white" alt="Lightning AI"></a>
  </p>

  <p>
    <a href="#-problem"><strong>Problem</strong></a> ·
    <a href="#-solution"><strong>Solution</strong></a> ·
    <a href="#-features"><strong>Features</strong></a> ·
    <a href="#-architecture"><strong>Architecture</strong></a> ·
    <a href="#-datasets"><strong>Datasets</strong></a> ·
    <a href="#-quick-start"><strong>Quick Start</strong></a> ·
    <a href="#-training-pipelines"><strong>Training</strong></a> ·
    <a href="#-roadmap"><strong>Roadmap</strong></a> ·
    <a href="#-license--disclaimer"><strong>License</strong></a>
  </p>

  <br>

  <img src="docs/assets/oralpath_pipeline.png" alt="OralPath Pipeline Overview" width="90%">
  <br><br>
  <p><em>📸 → 🧠 → 📊 — From microscope photo to structured clinical report</em></p>
</div>

---

## 🧬 Problem

Oral Squamous Cell Carcinoma (OSCC) accounts for **90% of oral cancers**, with **~377,000 new cases annually** worldwide. Survival rates exceed 80% when detected early but plummet below 30% in late stages. The challenge:

- **Pathologist shortage**: In low/middle-income countries, the pathologist-to-patient ratio can be **1:1,000,000+**
- **Delayed diagnoses**: Biopsies often take 2-4 weeks for processing and review
- **Subjective grading**: Inter-pathologist agreement on tumor grading is moderate (κ ≈ 0.5-0.7)

**OralPath is a non-commercial research project** that builds a **mobile-first AI diagnostic assistant** to address these gaps.

> ⚠️ **This is NOT a replacement for a pathologist.** It is a decision-support tool for resource-limited settings where dedicated oral pathology departments are unavailable. All outputs must be reviewed by a qualified pathologist before any clinical decision-making.

---

## 💡 Solution

A **multi-stage AI pipeline** that classifies and grades OSCC from H&E-stained biopsy slides photographed through a standard microscope:

| Stage | Task | Classes | Target |
|-------|------|---------|--------|
| **Stage 1** → | **Binary Detection** | Normal / OSCC | Sensitivity ≥ 0.95 |
| **Stage 2** → | **Grading** | Normal / OSMF / WD / MD / PD | Macro-F1 ≥ 0.65 |
| **Stage 3** *(v1.2)* → | **Segmentation** | Epithelium / Stroma / TILs / Collagen | mIoU ≥ 0.85 |

Results are surfaced through an **Android app** (Jetpack Compose) with structured reports, confidence scores, and full offline capability via TFLite.

---

## ✨ Features

<details open>
<summary><strong>🔬 AI Pipeline</strong></summary>

- **Frozen foundation backbones** — UNI (ViT-L/14), CTransPath, or EfficientNetB3 extract rich features without fine-tuning
- **Multi-source training** — Combines 5 public datasets (~2,084 cases) for robust generalization
- **Multiple Instance Learning (MIL)** — Whole-slide classification from patch embeddings using Attention-Top-K pooling
- **Stain normalization** — Macenko & Reinhard algorithms correct cross-lab H&E staining variation
- **Ordinal classification loss** — Exploits natural ordering of WD < MD < PD grades
- **Focal loss + weighted sampling** — Handles severe class imbalance in rare grades
</details>

<details>
<summary><strong>📱 Android App</strong></summary>

- **Jetpack Compose UI** — Modern, declarative UI with Material 3 theming
- **5-class result display** — Predicted class badge, confidence score, probability distribution bars
- **CameraX integration** *(planned)* — Capture slide photos directly from app
- **Room/SQLite storage** *(planned)* — Patient case history locally
- **PDF report export** *(planned)* — Structured clinical report with disclaimer
- **TFLite INT8 on-device** *(planned)* — Offline inference without connectivity
</details>

<details>
<summary><strong>🚀 Training Infrastructure</strong></summary>

- **Google Colab** — VS Code extension for GPU training with automatic dataset setup
- **Kaggle Kernels** — Pre-submitted kernels with bundled data; GPU probe detects T4/P100
- **Lightning AI Studio** — Persistent cloud GPU with multi-session resumable training
- **PowerShell automation** — One-command scripts for submit/monitor/download across all runtimes
- **Resumable checkpoints** — Training can pause/resume across Colab session limits
</details>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     📱 Android App                          │
│  Kotlin · Jetpack Compose · Coil · Room · CameraX          │
│  ┌────────┐   ┌──────────┐   ┌────────┐   ┌─────────────┐ │
│  │Camera  │──▶│View/Edit │──▶│Infer   │──▶│Results + PDF│ │
│  │Capture │   │Slide Img │   │(API/TFL)│   │Report       │ │
│  └────────┘   └──────────┘   └────────┘   └─────────────┘ │
│                                      │                      │
└──────────────────────────────────────┼──────────────────────┘
                                       │ REST / ONNX / TFLite
┌──────────────────────────────────────┼──────────────────────┐
│                🧠 Inference Server   │                      │
│  FastAPI · ONNX Runtime · Python     │                      │
│  ┌──────────────┐   ┌──────────────────────────┐           │
│  │  Preprocess   │──▶│   Stage 1 → Stage 2 →    │           │
│  │(Normalize ·   │   │    Detection + Grading    │           │
│  │ Resize ·      │   │    Ensemble               │           │
│  │ Patch Extract)│   └──────────────────────────┘           │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┼──────────────────────┐
│               🏋️ Training Pipeline   │                      │
│  PyTorch · timm · transformers       │                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │Stage 1   │   │Stage 2   │   │Stage 3   │               │
│  │Detection │   │Grading   │   │Segment   │ (v1.2)        │
│  └──────────┘   └──────────┘   └──────────┘               │
│  Colab · Kaggle · Lightning AI · Local GPU                │
└─────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
📁 oralpath/
├── 📱 android/               # Kotlin Android app (Jetpack Compose)
│   ├── app/src/main/java/com/oralpath/
│   │   ├── MainActivity.kt
│   │   ├── ui/result/        # Level 1 result screen
│   │   └── ui/theme/         # Material 3 color/typography
│   ├── build.gradle.kts
│   └── settings.gradle.kts
│
├── 🧠 model/                  # Python ML pipeline
│   ├── config.py              # Centralized paths and hyperparameters
│   ├── data/
│   │   ├── datasets/          # Multi-source dataset registry (5 sources)
│   │   │   ├── orchid_source.py, gdc_source.py
│   │   │   ├── multi_oscc_source.py, ndb_ufes_source.py
│   │   │   ├── wsi_patch_extraction.py
│   │   │   └── orchestrator.py  # Unified CLI entry point
│   │   ├── preprocessing/     # Dataset loader + zip-backed I/O
│   │   └── qa/                # Split builders, stain norm, integrity checks
│   ├── training/
│   │   ├── stage1_detection/  # Binary OSCC/Normal classifier
│   │   ├── stage2_grading/    # 5-class grading with resumable training
│   │   └── mil/               # MIL production pipeline (Attention-Top-K)
│   ├── evaluation/            # Metrics + case-level evaluation
│   ├── inference/             # CLI inference wrapper (JSON contract)
│   ├── kaggle/                # Kaggle kernel source code + bundles
│   ├── external/              # CTransPath, ORCHID, OralPatho references
│   └── notebooks/             # Colab bootstrap notebook
│
├── 📄 docs/                   # Comprehensive documentation
├── 📜 scripts/                # PowerShell + Python automation
├── 🧪 tests/                  # pytest test suite
└── 📦 external/               # Git submodules (OralPatho reference)
```

---

## 📊 Datasets

OralPath combines **5 public datasets** totaling **~2,084 cases** — one of the most comprehensive OSCC training collections for a research project:

| Dataset | Type | Size | Grade Labels | Use | License |
|---------|------|------|-------------|-----|---------|
| **Kaggle OSCC** | Patches | 1,224 (230 patients) | Normal / OSCC | Stage 1 binary | CC BY 4.0 |
| **ORCHID** | Patches | 23,000+ | Normal / OSMF / WD / MD / PD | Stage 2 grading | CC BY 4.0 |
| **TCGA-OSCC** | WSIs | ~257 | Grade info | MIL (Stage 2+) | GDC Open |
| **CPTAC-OSCC** | WSIs | ~165 | Grade info | MIL (Stage 2+) | GDC Open |
| **Multi-OSCC** | WSIs | 1,325 | Grade info | MIL (Stage 2+) | CC BY-SA 4.0 |
| **NDB-UFES** | Patches | 3,763 | Tissue type | Segmentation (v1.2) | CC BY 4.0 |

### Foundation Models

| Model | Role | License |
|-------|------|---------|
| **UNI** (Mahmood Lab, ViT-L/14) | Primary backbone | CC BY-NC-ND 4.0 |
| **CTransPath** | Secondary backbone | GPLv3-NC |
| **CONCH** (Mahmood Lab) | Research benchmark | CC BY-NC-ND 4.0 |
| **EfficientNetB3** | Fallback / benchmark | Apache 2.0 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- [Android Studio](https://developer.android.com/studio) (for mobile app development)

### 1. Set Up Python Environment

```bash
# Clone the repo
git clone https://github.com/ORION2809/OSCC.git
cd oralpath

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Unified dataset download + manifest generation
python scripts/download_datasets.py

# Verify manifests (dry run — no extraction needed)
python model/data/preprocessing/dataset_loader.py
```

### 3. Run Inference

```bash
python model/inference/stage2_predict.py \
    --image path/to/slide_image.png \
    --checkpoint path/to/stage2_checkpoint.pt
```

### 4. Build Android App

```bash
# Open android/ in Android Studio
# Sync Gradle → Select device/emulator → Run
```

---

## 🏋️ Training Pipelines

### 🧪 Local / Desktop GPU

```bash
# Stage 1 — Binary Detection (OSCC vs Normal)
python model/training/stage1_detection/train.py \
    --config model/training/stage1_detection/config.yaml

# Stage 2 — 5-Class Grading
python model/training/stage2_grading/train.py \
    --config model/training/stage2_grading/config.yaml
```

### ☁️ Google Colab (from VS Code)

```bash
# 1. Install the "Google Colab" VS Code extension
# 2. Open model/notebooks/oralpath_colab_bootstrap.ipynb
# 3. Click "Connect to Colab Runtime" — select a GPU backend
# 4. Run cells to mount Drive, clone repo, install deps, launch training
```

📖 See [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md) for the full VS Code + Colab workflow.

### 🏎️ [Kaggle Kernels](https://www.kaggle.com/)

```powershell
# Submit Stage 2 training kernel
.\scripts\run_kaggle_stage2.ps1

# Monitor progress
.\scripts\kaggle_stage2_status.ps1

# Download results when complete
.\scripts\download_kaggle_stage2_outputs.ps1
```

### ⚡ [Lightning AI Studio](https://lightning.ai/) (Recommended for Production)

```powershell
# Set your SSH target
$env:LIGHTNING_SSH_TARGET = "s_xxx@ssh.lightning.ai"

# Launch production MIL training
.\scripts\run_lightning_mil_production.ps1

# Monitor
.\scripts\lightning_mil_status.ps1
```

📖 See [`docs/LIGHTNING_TRAINING_HANDOFF.md`](docs/LIGHTNING_TRAINING_HANDOFF.md) for the canonical training guide.

---

## 🤖 Multi-Source MIL Pipeline

The most advanced experiment in the project uses **Multiple Instance Learning** for case-level classification:

1. **Embedding Extraction** — UNI backbone produces 1024-dim patch embeddings from 5 data sources
2. **Top-K Attention Pooling** — Learns to weight the most diagnostically-relevant patches
3. **Focal Loss** — Handles class imbalance across grades
4. **5-Fold Stratified Cross-Validation** — Per-source disaggregated metrics track generalization

```python
# Extract embeddings
python model/training/mil/extract_embeddings.py

# Train MIL model
python model/training/mil/train_mil_v2.py \
    --config model/training/mil/config.lightning_mil_production.yaml
```

---

## 🧪 Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run specific test modules
pytest tests/test_multi_source_pipeline.py -v
pytest tests/test_data_qa.py -v
pytest tests/test_mil_port.py -v
```

---

## 📈 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Stage 1 (Binary Detection)** | ✅ Prototype | Trained as background artifact |
| **Stage 2 (Grading)** | 🔄 Iterating | Run B: macro-F1 0.48; targeting ≥ 0.65 |
| **Multi-source MIL** | 🟢 Pipeline ready | Awaiting production run on Lightning AI |
| **Stage 3 (Segmentation)** | ⏳ Deferred | Target v1.2 |
| **Android App** | 🟡 Level 1 skeleton | Mock result screen complete |
| **Inference API** | 📝 Spec complete | JSON contract defined in docs |
| **ONNX/TFLite Export** | ⏳ Planned | |
| **Kaggle Integration** | ⚠️ Paused | GPU detection bug; Lightning AI preferred |

### Latest Training Results (Stage 2 — Run B)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.65 | 0.74 | 0.70 |
| OSMF | 0.41 | 0.48 | 0.44 |
| WD-OSCC | 0.47 | 0.57 | 0.51 |
| MD-OSCC | 0.24 | 0.12 | 0.16 |
| PD-OSCC | 0.56 | 0.52 | 0.54 |
| **Macro Avg** | **0.47** | **0.49** | **0.48** |

*Current focus: Improving MD-OSCC recall via focal loss, weighted sampling, and ordinal loss.*

---

## 🗺 Roadmap

```
v0.1 ── Research scaffold (this release)
│      • Repository structure, model interfaces, Android skeleton
│      • Multi-source dataset pipeline (5 sources)
│      • MIL production pipeline (Attention-Top-K)
│      • Kaggle + Colab + Lightning AI training automation
│
v1 ──── Detection + grading pilot
│      • Stage 1 binary model (sensitivity ≥ 0.95)
│      • Stage 2 grading model (macro-F1 ≥ 0.65)
│      • Android: camera capture, inference display, case history
│      • ONNX export + FastAPI inference server
│
v1.2 ── Segmentation
│      • Stage 3: MobileViT for tissue component ID
│      • TILs analysis, tumor-stroma ratio
│
v2 ──── Clinical pilot
       • Multi-centric validation study
       • Regulatory pathway assessment (IRB, CE/FDA)
       • On-device TFLite INT8 inference
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture, inference contract JSON, API spec |
| [`VISION.md`](docs/VISION.md) | Product vision, problem statement, target users, success metrics |
| [`MODEL_CARD.md`](docs/MODEL_CARD.md) | Model descriptions, intended use, limitations, fairness |
| [`DATASETS.md`](docs/DATASETS.md) | Complete dataset inventory, licenses, download instructions |
| [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Workstreams, acceptance gates, timeline |
| [`COLAB_SETUP.md`](docs/COLAB_SETUP.md) | VS Code + Colab training workflow |
| [`KAGGLE_STAGE2_TRAINING.md`](docs/KAGGLE_STAGE2_TRAINING.md) | Kaggle kernel submission runbook |
| [`LIGHTNING_TRAINING_HANDOFF.md`](docs/LIGHTNING_TRAINING_HANDOFF.md) | Canonical Lightning AI training guide |
| [`ORALPATHO_ADAPTATION.md`](docs/ORALPATHO_ADAPTATION.md) | Notes on adapting OralPatho reference architecture |
| [`DATA_QA_REBUILD_REPORT.md`](docs/DATA_QA_REBUILD_REPORT.md) | Dataset quality assurance and rebuild |

---

## 🤝 Contributing

This is a **non-commercial research project**. Contributions are welcome in the spirit of open science:

1. **Fork** the repo
2. **Create a feature branch:** `git checkout -b feature/my-idea`
3. **Commit your changes:** `git commit -m 'Add my idea'`
4. **Push:** `git push origin feature/my-idea`
5. **Open a Pull Request**

Areas where help is especially valuable:
- Improving MD-OSCC grading recall
- Adding new dataset sources
- Android UI development
- ONNX/TFLite export and optimization
- Documentation and test coverage

---

## 🙏 Acknowledgments

- **OralPatho** — Reference architecture adapted for MIL (MIT license) — [Repository](https://github.com/piyushpathology/oralpatho)
- **Mahmood Lab** — UNI and CONCH foundation models — [UNI](https://huggingface.co/mahmoodlab/UNI) | [CONCH](https://huggingface.co/mahmoodlab/CONCH)
- **CTransPath** — Transformer-based pathological feature extractor — [Repository](https://github.com/Xiyue-Wang/TransPath)
- **Dataset authors** — Kaggle OSCC (Tabassum et al.), ORCHID (NishaChaudhary23), TCGA, CPTAC, Multi-OSCC (Cavalcante et al.), NDB-UFES
- **Google Colab**, **Kaggle**, and **Lightning AI** for GPU compute resources

---

## 📄 License & Disclaimer

### License
- **Application code**: MIT or Apache 2.0 (to be finalized)
- **Models & datasets**: Varies per component — see [`docs/DATASETS.md`](docs/DATASETS.md) for per-component licensing
- **Project status**: Non-commercial research. NC-licensed models (UNI, CONCH, CTransPath) are used under their research terms

### Disclaimer
> **This application is intended for RESEARCH USE ONLY.**
>
> It does not provide a definitive medical diagnosis. The software is not cleared or approved by the FDA, CE, or any other regulatory body for clinical use. All outputs must be reviewed by a qualified pathologist before any clinical decision-making. The developers assume no liability for any clinical decisions made based on this software.

---

<div align="center">
  <sub>Built with ❤️ for open-source pathology AI research</sub>
  <br>
  <a href="https://github.com/ORION2809/OSCC/issues">Report Bug</a> ·
  <a href="https://github.com/ORION2809/OSCC/issues">Request Feature</a> ·
  <a href="https://github.com/ORION2809/OSCC/discussions">Join Discussion</a>
</div>
