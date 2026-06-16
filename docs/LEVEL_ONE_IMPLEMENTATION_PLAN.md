# OralPath Level 1 Implementation Plan

## Decision

Level 1 is not the normal-vs-OSCC detector.

The normal-vs-OSCC Stage 1 training can finish in the background, but it should not drive the first product workflow yet. For Level 1, the app should directly answer the question the user actually cares about:

```text
What class does this oral histopathology image belong to?
```

Level 1 classes:

```text
normal
osmf
wdoscc
mdoscc
pdoscc
```

Where:

- `normal` = no malignant/premalignant pattern detected in the patch
- `osmf` = oral submucous fibrosis / premalignant finding
- `wdoscc` = well differentiated oral squamous cell carcinoma
- `mdoscc` = moderately differentiated oral squamous cell carcinoma
- `pdoscc` = poorly differentiated oral squamous cell carcinoma

## Product Shape

Level 1 is a five-class patch/image classifier backed by ORCHID and a frozen pathology foundation model.

The app should not first show a binary "OSCC vs normal" result. Instead, the first usable output should be:

```json
{
  "label": "normal|osmf|wdoscc|mdoscc|pdoscc",
  "confidence": 0.0,
  "class_probabilities": {
    "normal": 0.0,
    "osmf": 0.0,
    "wdoscc": 0.0,
    "mdoscc": 0.0,
    "pdoscc": 0.0
  },
  "model_version": "stage2_grading_v1",
  "disclaimer": "research_use_only"
}
```

The existing Stage 1 normal-vs-OSCC model remains useful later as:

- a safety gate
- a fallback binary screen
- a triage/quality comparison model
- a possible ensemble input

But it is not the Level 1 user-facing model.

## Current Assets

Already present:

- ORCHID processed dataset at `model/data/processed/orchid`
- ORCHID manifest at `model/data/manifests/orchid.json`
- Stage 2 config at `model/training/stage2_grading/config.yaml`
- Stage 2 training script at `model/training/stage2_grading/train.py`
- UNI loading path already proven during Stage 1
- Colab CLI + Hugging Face + Kaggle credential path already proven
- Robust resumable Colab pattern from Stage 1

Important mismatch:

- Stage 1 has resumable chunk training.
- Stage 2 does not yet have resumable chunk training.

So the first Level 1 implementation task is not app UI. It is making Stage 2 trainable/restartable under the same unreliable Colab conditions.

Important training risk:

- OSMF is premalignant, not frank OSCC, and may overlap histologically with early WD-OSCC.
- OSMF must be watched as a first-class failure mode, not treated as an ordinary extra class.
- Before Stage 2 training starts, the ORCHID class distribution must be checked.
- If any class is meaningfully underrepresented, use weighted loss or class-balanced sampling from the first real run.
- In the current processed ORCHID split, OSMF is not the smallest class, but normal and PD-OSCC have fewer samples and still need imbalance handling.

## Level 1 Goal

Build and validate a five-class classifier that can classify a single microscopy patch/image into:

```text
normal / osmf / wdoscc / mdoscc / pdoscc
```

Level 1 is complete only when:

- Stage 2 trains to completion or early stopping.
- The report includes per-class precision, recall, F1, macro F1, weighted F1, and confusion matrix.
- No class collapses to near-zero recall.
- The model can be invoked through a stable local inference interface.
- Android can display the top label, confidence, and all class probabilities.

## Workstream 1: Freeze Stage 1 As Background

### Task 1.1: Let Stage 1 Finish, But Do Not Productize It

Keep the current Stage 1 training/watchdog running until epoch 50 or early stop.

Do not wire Stage 1 into Android as the primary result.

Stage 1 outputs should be archived as research artifacts:

- `model/evaluation/reports/colab_cli/stage1_report.json`
- `model/evaluation/reports/colab_cli/stage1_last.pt`
- final `stage1_best.pt` only when needed

Acceptance gate:

- Stage 1 status is documented, but no Level 1 UI depends on it.

## Workstream 2: Make Stage 2 Resumable

### Task 2.1: Add Stage 2 Resume Support

Mirror the Stage 1 resumable design in `model/training/stage2_grading/train.py`.

Required CLI args:

```text
--max-epochs
--resume-state
--state-output
```

Required checkpoint files:

```text
model/training/stage2_grading/checkpoints/stage2_last.pt
model/training/stage2_grading/checkpoints/stage2_best.pt
model/training/stage2_grading/logs/stage2_report.json
```

The resumable `stage2_last.pt` should contain:

- classifier head weights
- optimizer state
- scheduler state
- current epoch
- best validation metric
- history
- class names
- backbone name

Acceptance gate:

- A local or Colab smoke run can train one epoch, save `stage2_last.pt`, resume, and continue at the next epoch.

### Task 2.2: Add Batch Heartbeats

Stage 2 must print progress during each epoch:

```text
Epoch 012 batch 25/...
Epoch 012 batch 50/...
```

Acceptance gate:

- The Colab log clearly shows whether training is active or stale.

## Workstream 3: Stage 2 Colab Automation

### Task 3.1: Create Stage 2 Colab Job

Create:

```text
scripts/colab_cli_stage2_job.py
scripts/run_colab_cli_stage2.ps1
scripts/run_colab_cli_stage2_chunks.ps1
scripts/watch_colab_stage2_training.ps1
```

Use the Stage 1 pattern:

- upload Hugging Face token
- clone/update repo on Colab
- install dependencies once per session
- verify ORCHID dataset
- upload `stage2_last.pt` when present
- run exactly one resumable epoch per chunk
- download `stage2_report.json` and `stage2_last.pt`
- skip huge full checkpoint download except at final export time
- watchdog restarts stale runners

Acceptance gate:

- `run_colab_cli_stage2_chunks.ps1 -TargetEpoch 50` can survive Colab pruning and advance one epoch at a time.

## Workstream 4: Train Level 1 Model

### Task 4.0: Dataset Distribution Preflight

Before training, print and save ORCHID sample counts by split and class:

```text
train: normal / osmf / wdoscc / mdoscc / pdoscc
val:   normal / osmf / wdoscc / mdoscc / pdoscc
test:  normal / osmf / wdoscc / mdoscc / pdoscc
```

Current processed ORCHID counts:

| Split | normal | osmf | wdoscc | mdoscc | pdoscc |
|---|---:|---:|---:|---:|---:|
| train | 1,045 | 2,095 | 2,790 | 2,699 | 1,599 |
| val | 294 | 592 | 788 | 760 | 451 |
| test | 163 | 328 | 433 | 421 | 247 |

Training policy:

- Treat OSMF recall and OSMF-vs-WD confusion as a primary quality signal.
- Enable weighted loss or a weighted sampler before full training if the max/min class ratio is high enough to bias learning.
- Do not wait for OSMF, normal, or PD-OSCC recall to collapse before adding imbalance handling.

Acceptance gate:

- The Stage 2 report records class counts and whether weighting/sampling was enabled.
- The first full run is configured with imbalance handling when class counts justify it.

### Task 4.1: Stage 2 Smoke Test

Run:

```powershell
python model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml --dry-run
python model/training/stage2_grading/train.py --config model/training/stage2_grading/config.yaml --max-batches 1 --max-epochs 1
```

Acceptance gate:

- ORCHID class counts load correctly.
- UNI loads.
- One batch trains.
- One validation pass runs.
- One test pass runs.

### Task 4.2: Full Stage 2 Training

Run chunked Colab training:

```powershell
.\scripts\run_colab_cli_stage2_chunks.ps1 -TargetEpoch 50
```

Do not treat 50 epochs as mandatory. With UNI/CTransPath frozen, only the classification head is learning, so useful convergence may happen around epoch 15-20.

Training policy:

- Review validation macro F1, OSMF recall, and per-class F1 after epochs 5, 10, 15, and 20.
- Stop early if validation macro F1 has clearly plateaued and per-class recall is stable.
- Continue toward 50 only if the validation curve is still improving or if minority-class recall is still recovering.
- Keep early stopping enabled so chunked Colab training does not blindly burn runtime.

Primary metric:

```text
macro_f1
```

Secondary metrics:

```text
weighted_f1
overall accuracy
per-class recall
per-class F1
confusion matrix
```

Acceptance gate:

- `stage2_report.json` exists and includes all five classes.
- `macro_f1` is not dominated by normal or one easy class.
- WD/MD/PD confusion is visible and documented.

## Workstream 5: Level 1 Quality Gate

The Level 1 model is not accepted just because training finishes.

Minimum quality gate:

```text
macro_f1 >= 0.70 target
weighted_f1 >= 0.75 target
normal recall >= 0.75 target
osmf recall >= 0.65 target
wdoscc recall >= 0.65 target
mdoscc recall >= 0.65 target
pdoscc recall >= 0.65 target
```

If the model misses the target:

1. Inspect confusion matrix.
2. Check class imbalance.
3. Add weighted loss or sampler.
4. Try lower learning rate for classifier head.
5. Try EfficientNetB3 baseline.
6. Try CTransPath if weights are available.
7. Consider three-class OSCC grading only: `wdoscc/mdoscc/pdoscc`, with `normal/osmf` handled separately.

## Workstream 6: Inference Contract

### Task 6.1: Create Level 1 Inference Wrapper

Create a Python inference entry point:

```text
model/inference/stage2_predict.py
```

Input:

```text
image path
checkpoint path
```

Output:

```json
{
  "label": "wdoscc",
  "confidence": 0.82,
  "class_probabilities": {
    "normal": 0.01,
    "osmf": 0.03,
    "wdoscc": 0.82,
    "mdoscc": 0.12,
    "pdoscc": 0.02
  },
  "model_version": "stage2_grading_v1",
  "disclaimer": "research_use_only"
}
```

Acceptance gate:

- A single local ORCHID test image produces a valid JSON response.

## Workstream 7: Android Level 1 Vertical Slice

### Task 7.1: UI Result Screen

The first app screen after image selection should show:

- predicted class
- confidence
- probability bars for all five classes
- research-use disclaimer
- image thumbnail

Do not show Stage 1 binary result in Level 1.

### Task 7.2: Mock Then Real Inference

Phase A:

- Android uses a mocked Level 1 JSON response.

Phase B:

- Android consumes the Python/API inference output.

Phase C:

- Export model to ONNX/TFLite only after metrics are acceptable.

Acceptance gate:

- A user can select/capture an image and see one of the five Level 1 classes.

## Workstream 8: Reporting

The Level 1 report should include:

- case ID
- image thumbnail
- predicted class
- confidence
- all class probabilities
- model version
- research-use disclaimer

Do not claim diagnostic certainty.

Do not include Stage 1 output unless explicitly added later as supporting signal.

## Execution Order

1. Let Stage 1 finish in background.
2. Add Stage 2 resumable training.
3. Add Stage 2 Colab automation/watchdog.
4. Run Stage 2 smoke test.
5. Train Stage 2 to completion.
6. Evaluate per-class quality.
7. Build Level 1 inference wrapper.
8. Build Android mock result screen.
9. Connect real Stage 2 inference.
10. Decide whether Stage 1 should be reintroduced as a later safety gate.

## Definition Of Done

Level 1 is done when:

- The app can classify an image into `normal/osmf/wdoscc/mdoscc/pdoscc`.
- The result screen does not depend on the binary Stage 1 model.
- The Stage 2 model has a documented report with per-class metrics.
- The model artifact, class order, preprocessing, and output JSON are versioned.
- The app clearly states research-use-only.
