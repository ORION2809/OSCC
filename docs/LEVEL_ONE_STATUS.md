# OralPath Level 1 Implementation Status

> Last updated: 2026-06-16

## Summary

Level 1 implementation is code-complete for the training path and ready for full
GPU training on Colab. Stage 1 remains a background research artifact and is not
used as the Level 1 product gate.

## Implemented

### Stage 2 Resumable Training

- `model/training/stage2_grading/train.py` supports:
  - `--max-epochs`
  - `--resume-state`
  - `--state-output`
  - batch heartbeats every 25 batches
- `stage2_last.pt` stores classifier-head state, optimizer state, scheduler
  state, epoch, best macro F1, history, class names, and backbone name.
- `stage2_best.pt` stores the best full model checkpoint by validation macro F1.

### ORCHID Preflight And Imbalance Handling

- Training prints train/val/test counts for all five classes before fitting.
- `stage2_report.json` records class distribution and imbalance settings.
- `config.yaml` supports `none`, `weighted_loss`, and `weighted_sampler`.
- The default is `weighted_loss` because the current ORCHID train max/min ratio
  is 2.67.

### Colab Automation

- `scripts/colab_cli_stage2_job.py`
- `scripts/run_colab_cli_stage2.ps1`
- `scripts/run_colab_cli_stage2_chunks.ps1`
- `scripts/watch_colab_stage2_training.ps1`

The Colab path uses temporary `/content` storage, downloads ORCHID only when
needed, resumes from `stage2_last.pt`, and downloads the report plus resumable
state after each chunk.

### Inference Contract

- `model/inference/stage2_predict.py` returns:
  - `label`: `normal|osmf|wdoscc|mdoscc|pdoscc`
  - `confidence`
  - `class_probabilities`
  - `model_version`
  - `disclaimer`: `research_use_only`

## Local Verification

Passed:

```powershell
.venv\Scripts\python.exe model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml --dry-run
.venv\Scripts\python.exe model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml --max-batches 1 --max-epochs 1
py -3 -m py_compile model/training/stage2_grading/train.py scripts/colab_cli_stage2_job.py scripts/setup_colab_ephemeral_data.py model/inference/stage2_predict.py
```

The one-batch smoke run loaded UNI, applied weighted loss, saved
`stage2_last.pt`, and wrote `stage2_report.json`. Smoke-run metrics are not
meaningful because only one batch was used.

Not run:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Reason: `pytest` is not installed in the local venv.

## Next Step

Start full Stage 2 training with the watchdog:

```powershell
.\scripts\watch_colab_stage2_training.ps1 -TargetEpoch 50 -CheckSeconds 300
```

Review validation macro F1, OSMF recall, and per-class F1 at epochs 5, 10, 15,
and 20. Stop before epoch 50 if validation macro F1 has plateaued and all
minority-class recall values are stable.
