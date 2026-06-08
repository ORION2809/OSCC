# OralPath — Dataset and Model Source Inventory

> Non-commercial research project. All assets used must comply with their respective licenses.

---

## Datasets

### 1. Kaggle OSCC (Tabassum et al., 2020)

| Attribute | Value |
|---|---|
| **Role** | Model 1 (binary detection) training / validation |
| **Images** | 1,224 (290 normal, 934 cancerous) |
| **Patients** | 230 |
| **Magnifications** | 100× and 400× |
| **License** | CC BY 4.0 |
| **Source** | [Kaggle](https://www.kaggle.com/datasets) — search "OSCC histopathological" |
| **Local path convention** | `model/data/raw/kaggle_oscc/` |
| **Expected structure** | `normal/`, `oscc/` subfolders |

**Notes**
- Used for Stage 1: OSCC vs Normal binary classification.
- All images are H&E stained.
- Patch extraction may be applied to increase training sample count.

---

### 2. ORCHID Dataset (NishaChaudhary23 et al., 2024)

| Attribute | Value |
|---|---|
| **Role** | Model 2 (grading) training / validation |
| **Images** | 23,000+ patches |
| **Centers** | Multi-center Indian dataset |
| **Grades** | Normal / OSMF / WD / MD / PD |
| **License** | CC BY 4.0 |
| **Source** | [Hugging Face / ORCHID](https://huggingface.co/datasets) or publication supplementary |
| **Local path convention** | `model/data/raw/orchid/` |
| **Expected structure** | `normal/`, `osmf/`, `wd/`, `md/`, `pd/` subfolders |

**Notes**
- Built by the same author as OralPatho.
- Five-class split is natively supported; training scripts are available.
- OSMF inclusion is a v1 decision gate — ORCHID supports it with clean labels.

---

### 3. NDB-UFES

| Attribute | Value |
|---|---|
| **Role** | Model 3 (segmentation) training — deferred to v1.2 |
| **Images** | 3,763 patches (512×512) |
| **Annotations** | OSCC / dysplasia / normal with epithelial / stromal masks |
| **License** | Open access — verify exact terms before deployment |
| **Source** | Academic repository (search "NDB-UFES oral cancer") |
| **Local path convention** | `model/data/raw/ndb_ufes/` |

**Notes**
- Deferred to v1.2. Do not download until segmentation workstream begins.

---

## Foundation Models

### UNI (Harvard Mahmood Lab)

| Attribute | Value |
|---|---|
| **Role** | Primary frozen feature extractor candidate |
| **Training data** | 100M pathology images from 100,000 WSIs |
| **License** | CC-BY-NC-ND 4.0 |
| **Meaning for OralPath** | Fully permitted for non-commercial research |
| **Access** | [Hugging Face](https://huggingface.co/mahmoodlab/UNI) |
| **Embedding shape** | 1024-dim (ViT-L/14) |
| **Notes** | Do not fine-tune for v1. Use frozen backbone + trainable head only. |

---

### CTransPath

| Attribute | Value |
|---|---|
| **Role** | Primary frozen feature extractor candidate |
| **Training data** | 15M histopathology patches from 32 cancer subtypes |
| **License** | GPLv3-NC |
| **Meaning for OralPath** | Fully permitted for non-commercial research |
| **Access** | Academic repository / GitHub (search "CTransPath") |
| **Notes** | Strong pathology-specific pre-training. Evaluate against UNI. |

---

### CONCH (Harvard Mahmood Lab)

| Attribute | Value |
|---|---|
| **Role** | Research benchmark / comparison only |
| **Type** | Vision-language pathology model |
| **License** | CC-BY-NC-ND 4.0 |
| **Meaning for OralPath** | Research validation permitted; allows text queries |
| **Access** | [Hugging Face](https://huggingface.co/mahmoodlab/CONCH) |
| **Notes** | Benchmark in research environment. Not required for v1 deployment. |

---

### Path Foundation

| Attribute | Value |
|---|---|
| **Role** | Research benchmark candidate |
| **License** | NC research license |
| **Meaning for OralPath** | Permitted for non-commercial research |
| **Access** | Verify current availability and access constraints |
| **Notes** | Include in benchmark matrix if accessible. |

---

## Reference Architectures

### OralPatho

| Attribute | Value |
|---|---|
| **Role** | Model 1 and Model 2 architecture / training script reference |
| **Published** | 2025, medRxiv |
| **License** | MIT |
| **Training data** | 1,925 WSIs from five Indian hospitals |
| **Source** | GitHub repository (search "OralPatho OSCC") |
| **Local path** | `model/external/oralpatho/` |
| **Notes** | Clone and inventory. Adapt Stage 1 and Stage 2 for patch input. |

---

## License Inventory Summary

| Component | License | Research Use | Commercial Use |
|---|---|---|---|
| Kaggle OSCC dataset | CC BY 4.0 | ✅ Permitted | ✅ Permitted |
| ORCHID dataset | CC BY 4.0 | ✅ Permitted | ✅ Permitted |
| NDB-UFES dataset | Open access | ✅ Verify terms | ⚠️ Verify terms |
| OralPatho code | MIT | ✅ Permitted | ✅ Permitted |
| UNI weights | CC-BY-NC-ND 4.0 | ✅ Permitted | ❌ Prohibited |
| CONCH weights | CC-BY-NC-ND 4.0 | ✅ Permitted | ❌ Prohibited |
| CTransPath weights | GPLv3-NC | ✅ Permitted | ❌ Prohibited |
| Path Foundation | NC research | ✅ Permitted | ❌ Prohibited |
| EfficientNetB3 | Apache 2.0 | ✅ Permitted | ✅ Permitted |
| MobileViT | Apache 2.0 | ✅ Permitted | ✅ Permitted |
| OralPath app code | MIT / Apache 2.0 (planned) | ✅ Permitted | ✅ Permitted |

---

## Download Checklist

- [ ] Kaggle OSCC dataset downloaded to `model/data/raw/kaggle_oscc/`
- [ ] ORCHID dataset downloaded to `model/data/raw/orchid/`
- [ ] OralPatho repository cloned to `model/external/oralpatho/`
- [ ] UNI model weights downloaded / cached
- [ ] CTransPath model weights downloaded / cached
- [ ] NDB-UFES **deferred** to v1.2

---

*Last updated: June 2026*
