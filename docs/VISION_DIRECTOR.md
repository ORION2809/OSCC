# OralPath Vision Director
we are building this for a non commercial purposes
## After — What The Architecture Actually Is Now

This document directs the current interpretation of `VISION.md`. The original vision, product scope, Android stack, success metrics, regulatory path, repository structure, and references remain valid. What changes is the AI architecture and training strategy.

The entire approach flips.

Instead of building upward from a general vision model, OralPath now stands on top of work that already exists and is specifically designed for this problem. The project is no longer trying to invent an OSCC architecture from scratch. It is adapting proven histopathology and OSCC-specific systems into a mobile-first research tool.

## Backbone

The backbone — the part that actually understands what histopathology tissue looks like — comes from UNI or CTransPath.

UNI was trained on 100 million pathology images from 100,000 whole-slide images. CTransPath was trained on 15 million histopathology patches from 32 cancer subtypes. Either one arrives already knowing what cell nuclei, stromal tissue, epithelial layers, tumor regions, and tissue morphology look like.

That backbone stays frozen.

Do not retrain UNI or CTransPath. Do not fine-tune the foundation model unless a later validation phase proves it is necessary. For v1, the training only adjusts the small classification heads on top.

The practical architecture is:

| Model | License | Role |
|---|---|---|
| UNI or CTransPath | CC-BY-NC-ND / GPLv3-NC | Primary frozen feature extractor backbone |
| EfficientNetB3 | Apache 2.0 | Fallback and benchmarking comparison |
| MobileViT | Apache 2.0 | Model 3 segmentation backbone only |

Because OralPath is non-commercial research, NC-licensed models are allowed. No commercial license negotiation is needed for UNI, CONCH, CTransPath, or Path Foundation in this phase.

## Models 1 And 2

The training pipeline for Models 1 and 2 comes from OralPatho.

OralPatho is a system built specifically for OSCC. It was trained on 1,925 whole-slide images from five Indian hospitals, published in 2025, MIT licensed, and its code and training scripts are publicly available.

That matters.

OralPath is not designing a new architecture in isolation. It is adapting an architecture that already works on Indian patient data, which is directly relevant to the intended deployment context.

Model 1 should adapt OralPatho Stage 1:

| Stage | Purpose | Adaptation |
|---|---|---|
| Model 1 | OSCC vs normal detection | Clone OralPatho, adapt Stage 1 to patch input, train on Kaggle OSCC patches |

Model 2 should adapt OralPatho Stage 2:

| Stage | Purpose | Adaptation |
|---|---|---|
| Model 2 | Grading and class prediction | Adapt OralPatho Stage 2 using ORCHID scripts, train on ORCHID patches |

The dataset for Model 2 comes from ORCHID, built by the same author as OralPatho, `NishaChaudhary23`. ORCHID uses a CC BY 4.0 license and already includes training scripts for the exact five-class split OralPath should care about:

- Normal
- OSMF
- Well-differentiated OSCC
- Moderately-differentiated OSCC
- Poorly-differentiated OSCC

This means OSMF should be treated as a serious v1 decision, not a distant future feature. OralPatho and ORCHID both support it natively. Adding it during training costs very little and strengthens the research and grant case substantially.

## Model 3

Model 3 is segmentation.

This does not belong in v1.

Model 3 uses MobileViT as its backbone. MobileViT is Apache 2.0 licensed, designed for mobile inference, and the architecture is fully described in the OralTILs-ViT paper.

The implementation plan is:

| Model | Purpose | Architecture | Dataset | Phase |
|---|---|---|---|---|
| Model 3 | Tissue / TIL segmentation | TILSeg-MobileViT | NDB-UFES | v1.2 |

Implement TILSeg-MobileViT once from the OralTILs-ViT paper and train it on NDB-UFES. Keep this out of the first release so v1 can focus on detection, grading, capture quality, reporting, and validation.

## Training Plan Override

Read Section 6 of `VISION.md` through this corrected plan:

1. Environment setup stays.
2. Dataset preparation stays.
3. Stage 1 changes from fine-tuning EfficientNetB3 to adapting OralPatho Stage 1 with a frozen UNI or CTransPath backbone.
4. Stage 2 changes from a generic multi-class head to adapting OralPatho Stage 2 using ORCHID training scripts.
5. Stage 3 changes from DeepLabv3+ to TILSeg-MobileViT and moves to v1.2.
6. Export still targets ONNX and TFLite.
7. Benchmarking must include EfficientNetB3, CTransPath, UNI, CONCH, Path Foundation, and original OralPatho results.

## License Strategy Override

Read Section 7 of `VISION.md` through this corrected license table:

| Component | License | Meaning For OralPath |
|---|---|---|
| UNI / CONCH | CC-BY-NC-ND 4.0 | Fully permitted for non-commercial research |
| CTransPath | GPLv3-NC | Fully permitted for non-commercial research |
| Path Foundation | NC research license | Fully permitted for non-commercial research |
| OralPatho code | MIT | Architecture and training scripts reference |
| ORCHID dataset | CC BY 4.0 | Training data for Model 2 |
| Kaggle OSCC dataset | CC BY 4.0 | Training data for Model 1 |
| EfficientNetB3 | Apache 2.0 | Fallback and benchmark |
| MobileViT | Apache 2.0 | Segmentation backbone for v1.2 |

Delete the idea of negotiating a UNI or CONCH commercial license from the current plan. It is irrelevant for a free, non-commercial research project.

## Phase Override

Phase 0 should focus on cloning OralPatho, adapting Stage 1 to patch input, and training on the Kaggle OSCC dataset.

Phase 1 should focus on adapting OralPatho Stage 2 with ORCHID scripts and training on ORCHID patches.

Phase 2 should move Model 3 segmentation into v1.2, using TILSeg-MobileViT trained on NDB-UFES.

## Main Risk To Track

The obsolete risk is commercial license denial for UNI or CONCH.

Replace it with the real technical risk:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OralPatho WSI-to-patch adaptation fails to match published accuracy | Medium | Medium | Benchmark against original OralPatho results; fall back to EfficientNetB3 fine-tune if gap is greater than 5% |

## Bottom Line

OralPath is not a generic Android app wrapped around a newly trained EfficientNet model.

It is a mobile-first OSCC diagnostic research tool built by combining:

- Frozen pathology foundation backbones
- OralPatho's OSCC-specific architecture
- ORCHID's five-class patch dataset
- Kaggle OSCC detection data
- Deferred MobileViT segmentation for v1.2

That is the architecture now.
