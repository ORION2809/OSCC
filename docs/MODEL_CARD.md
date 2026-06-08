# OralPath — Model Card

> Non-commercial research project. This document describes the models used in OralPath v1.

---

## Model 1: OSCC Detection

| Attribute | Value |
|---|---|
| **Name** | oralpath-detection-v1 |
| **Version** | 1.0.0 |
| **Task** | Binary classification — OSCC vs Normal tissue |
| **Target users** | Doctors, dentists, pathologists in resource-limited settings |
| **Intended use** | Preliminary screening aid, not a replacement for pathologist review |

### Architecture

| Component | Value |
|---|---|
| **Backbone** | UNI (ViT-L/14) or CTransPath — frozen |
| **Head** | Single linear layer → sigmoid |
| **Input shape** | 224×224 or 300×300 RGB |
| **Output** | Binary label + confidence |

### Training Data

| Dataset | Images | Role |
|---|---|---|
| Kaggle OSCC (Tabassum et al.) | 1,224 (290 normal, 934 OSCC) | Training + validation |

### Performance Targets

| Metric | Target |
|---|---|
| Sensitivity | ≥ 0.95 |
| Specificity | ≥ 0.90 |
| AUC-ROC | Reported |
| F1 Score | Reported |

### Limitations

- Trained on H&E stained slides only; other stains will not perform well
- Designed for 100× and 400× magnification; other magnifications untested
- Performance may degrade on scanners / cameras not represented in training
- All outputs must be reviewed by a qualified pathologist

---

## Model 2: OSCC Grading

| Attribute | Value |
|---|---|
| **Name** | oralpath-grading-v1 |
| **Version** | 1.0.0 |
| **Task** | Multi-class classification — Normal / OSMF / WD / MD / PD |
| **Target users** | Same as Model 1 |
| **Intended use** | Grade estimation after OSCC detection; referral decision support |

### Architecture

| Component | Value |
|---|---|
| **Backbone** | Same frozen backbone as Model 1 |
| **Head** | Multi-class linear layer → softmax |
| **Input shape** | 224×224 or 300×300 RGB |
| **Output** | Class label + confidence |

### Training Data

| Dataset | Images | Role |
|---|---|---|
| ORCHID (NishaChaudhary23 et al.) | 23,000+ patches | Training + validation |

### Classes

| Label | Description |
|---|---|
| `normal` | Normal oral mucosa |
| `osmf` | Oral Submucous Fibrosis (pre-malignant) |
| `wd` | Well-differentiated OSCC |
| `md` | Moderately-differentiated OSCC |
| `pd` | Poorly-differentiated OSCC |

### Performance Targets

| Metric | Target |
|---|---|
| Overall accuracy | ≥ 0.85 |
| Per-class precision/recall/F1 | Reported |
| Confusion matrix | Reported |

### Limitations

- WD vs MD and MD vs PD are known difficult boundaries in histopathology
- OSMF classification is experimental and requires additional clinical validation
- Grading is inherently subjective; inter-pathologist agreement varies

---

## Model 3: Tissue Segmentation (v1.2)

| Attribute | Value |
|---|---|
| **Name** | oralpath-segmentation-v1 |
| **Version** | Deferred to v1.2 |
| **Task** | Semantic segmentation — epithelial, stromal, TILs, collagen |
| **Architecture** | TILSeg-MobileViT |
| **Training data** | NDB-UFES |
| **Target** | mIOU ≥ 0.85 |

---

## Ethical and Safety Considerations

1. **Research use only** — All outputs include a disclaimer that this is not a definitive diagnosis.
2. **Pathologist review required** — The app explicitly recommends review by a qualified pathologist.
3. **No cloud sync of patient data** — All images and results stay on-device by default.
4. **DPDPA 2023 compliant** — No personal health information leaves the device without explicit consent.
5. **Non-commercial** — Licensed models (UNI, CONCH, CTransPath) are used under their non-commercial terms.

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | June 2026 | Initial scaffold — models not yet trained |

---

*Last updated: June 2026*
