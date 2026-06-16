# Kaggle Stage 2 Training

Kaggle is now the preferred runtime for Level 1 Stage 2 training. Colab CLI is
kept as a fallback, but the ORCHID Zenodo download path was unreliable through
Colab.

## Why Kaggle

- Kaggle provides long-running background notebook execution.
- The ORCHID mirror already exists on Kaggle:
  `nazmulxdxd/orchid-oscc-classification`
- The dataset can be attached directly to the kernel, avoiding local upload and
  avoiding the Colab Zenodo download stall.

## One-Time Requirement

Add a Kaggle secret named:

```text
HF_TOKEN
```

The token must have access to the UNI model on Hugging Face. The Kaggle runner
will fail loudly if UNI cannot be loaded; it will not silently fall back to
EfficientNetB3.

## Submit Training

```powershell
.\scripts\run_kaggle_stage2.ps1
```

This pushes the private Kaggle script kernel:

```text
oralpath_user/oralpath-stage2-orchid-level1
```

The kernel attaches:

```text
nazmulxdxd/orchid-oscc-classification
```

## Check Status

```powershell
.\scripts\kaggle_stage2_status.ps1
.\scripts\kaggle_stage2_status.ps1 -Logs
```

## Download Outputs

```powershell
.\scripts\download_kaggle_stage2_outputs.ps1
```

Expected output files:

```text
model/evaluation/reports/kaggle_stage2/stage2_report.json
model/evaluation/reports/kaggle_stage2/stage2_last.pt
model/evaluation/reports/kaggle_stage2/stage2_best.pt
```

## Runtime Notes

- Training uses `model/training/stage2_grading/config.kaggle.yaml`.
- Dataset root is resolved from `/kaggle/input/orchid-oscc-classification`.
- The loader supports Kaggle's nested ORCHID structure:
  `ORCHID_train/train/<class>/<wsi>/<patch>.png`.
- Primary validation metric is macro F1.
- Watch validation macro F1, OSMF recall, and per-class F1 around epochs 5, 10,
  15, and 20.
