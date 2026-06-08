# Colab Training Commands

Run these in `model/notebooks/oralpath_colab_bootstrap.ipynb` after the notebook is connected to a Google Colab GPU runtime.

## 1. Runtime Check

```bash
python scripts/colab_runtime_check.py
```

Expected:

- `Colab module available: True`
- `CUDA available: True`
- A GPU name such as T4, L4, or A100

## 2. Dataset Dry Runs

```bash
python model/training/stage1_detection/train.py --config model/training/stage1_detection/config.yaml --dry-run
python model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml --dry-run
```

Expected local counts:

- Stage 1: train 4946, val 120, test 126
- Stage 2: train 10228, val 2885, test 1592

## 3. Smoke Runs

```bash
python model/training/stage1_detection/train.py --config model/training/stage1_detection/config.yaml --max-batches 1
python model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml --max-batches 1
```

These confirm that the selected backbone, transforms, labels, dataloaders, metrics, and checkpoint writing all work before committing to a long run.

## 4. Full Training

Run Stage 1 first:

```bash
python model/training/stage1_detection/train.py --config model/training/stage1_detection/config.yaml
```

Then run Stage 2:

```bash
python model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml
```

## 5. Outputs

Stage 1 writes:

- `model/training/stage1_detection/checkpoints/stage1_best.pt`
- `model/training/stage1_detection/logs/stage1_report.json`

Stage 2 writes:

- `model/training/stage2_grading/checkpoints/stage2_best.pt`
- `model/training/stage2_grading/logs/stage2_report.json`

Copy these to Google Drive after each successful run if the notebook does not already sync the repo output directory.
