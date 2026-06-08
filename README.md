# OralPath — OSCC Diagnostic Assistant

> **Non-commercial research project.** A mobile-first diagnostic assistant for Oral Squamous Cell Carcinoma (OSCC) histopathological slide analysis.

## What It Does

OralPath helps doctors analyze H&E stained biopsy slides photographed through a microscope:

1. **Capture** — Take a photo via the Android app or upload an existing slide image
2. **Analyze** — AI classifies tissue as Normal or OSCC, and grades cancer as Well-differentiated / Moderately-differentiated / Poorly-differentiated
3. **Report** — View confidence scores, attention heatmaps, and export a structured PDF report

This is **not a replacement for a pathologist**. It is a decision-support tool for resource-limited settings where dedicated oral pathology departments are unavailable.

## Repository Structure

```
oralpath/
├── android/              # Kotlin Android app (Jetpack Compose)
├── model/                # Python training + export pipeline
│   ├── data/             # Dataset manifests and preprocessing
│   ├── training/         # Stage 1/2/3 training scripts
│   ├── export/           # ONNX + TFLite export
│   ├── evaluation/       # Benchmarking and metrics
│   ├── notebooks/        # Colab / Jupyter notebooks
│   └── external/         # OralPatho, ORCHID reference code
├── docs/                 # Architecture, datasets, model cards
├── scripts/              # Helper scripts
└── tests/                # Test suite
```

## Key Documents

| Document | Purpose |
|---|---|
| [`docs/VISION.md`](docs/VISION.md) | Product vision, problem, users, success metrics |
| [`docs/VISION_DIRECTOR.md`](docs/VISION_DIRECTOR.md) | AI architecture direction — frozen backbones, OralPatho adaptation |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Step-by-step workstreams and acceptance gates |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture, inference contract JSON schema, API spec |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Dataset sources, licenses, download instructions |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model descriptions, limitations, ethical considerations |
| [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md) | VS Code + Google Colab training workflow |

## Quick Start

### Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify dataset manifests (dry run)
python model/data/preprocessing/dataset_loader.py

# Download instructions
python scripts/download_datasets.py
```

### Colab Training From VS Code

The official Google Colab VS Code extension is recommended for GPU training. Open [`model/notebooks/oralpath_colab_bootstrap.ipynb`](model/notebooks/oralpath_colab_bootstrap.ipynb), connect it to a Colab runtime, mount Google Drive, clone/update the repo, and run the training commands from there.

See [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md) for the full workflow.

### Android

```bash
# Open android/ in Android Studio
# Sync Gradle and run on emulator or device
```

## Version Scope

| Version | Goal |
|---|---|
| **v0.1** (current) | Research scaffold — repo structure, model interfaces, Android skeleton |
| **v1** | Detection + grading pilot — capture, inference, result display, case storage, PDF export |
| **v1.2** | Segmentation — tissue component identification with MobileViT |

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile app | Kotlin, Jetpack Compose, CameraX, Room |
| Architecture | MVVM + Clean Architecture, Hilt DI |
| ML backbones | UNI, CTransPath (frozen), EfficientNetB3 (fallback) |
| Training | PyTorch, timm, transformers |
| Export | ONNX (server), TFLite INT8 (on-device fallback) |
| API | FastAPI (local inference server) |

## License

- App code: MIT or Apache 2.0 (to be finalized)
- Models and datasets: See [`docs/DATASETS.md`](docs/DATASETS.md) for per-component license details
- This is a **non-commercial research project**. NC-licensed models (UNI, CONCH, CTransPath) are used under their research terms.

## Disclaimer

This application is intended for **research use only**. It does not provide a definitive medical diagnosis. All outputs must be reviewed by a qualified pathologist before any clinical decision-making.

---

*Last updated: June 2026*
