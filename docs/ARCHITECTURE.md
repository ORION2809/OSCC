# OralPath — Architecture Document

> Non-commercial research project. This document defines the system architecture and the inference contract between model training, API inference, and the Android application.

---

## 1. System Overview

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Android   │────▶│  Preprocessing      │────▶│  Inference      │
│   App       │     │  (on-device)        │     │  (API / Local)  │
│  (Kotlin)   │◀────│                     │◀────│                 │
└─────────────┘     └─────────────────────┘     └─────────────────┘
       │                                                │
       ▼                                                ▼
┌─────────────┐                               ┌─────────────────┐
│  Room DB    │                               │  Model Server   │
│  (SQLite)   │                               │  (FastAPI / HF) │
└─────────────┘                               └─────────────────┘
       │                                                │
       ▼                                                ▼
┌─────────────┐                               ┌─────────────────┐
│  PDF Export │                               │  PyTorch Models │
│  (local)    │                               │  (ONNX / TFLite)│
└─────────────┘                               └─────────────────┘
```

---

## 2. Android Architecture

| Layer | Technology |
|---|---|
| UI | Jetpack Compose |
| Architecture | MVVM + Clean Architecture |
| Dependency Injection | Hilt |
| Navigation | Jetpack Navigation Component |
| Camera | CameraX |
| Image Display | Glide |
| On-device ML | TFLite (fallback) |
| Networking | Retrofit 2 + OkHttp3 |
| Serialization | Kotlin Serialization |
| Local Database | Room (SQLite) |
| PDF Generation | Android PdfDocument API |
| Async | Kotlin Coroutines + Flow |
| Testing | JUnit 4, Espresso, MockK |

### Package Structure

```
com.oralpath/
├── ui/                 # Compose screens and ViewModels
│   ├── capture/
│   ├── result/
│   ├── history/
│   └── report/
├── domain/             # UseCases and business logic
│   ├── model/          # Domain models
│   └── usecase/        # UseCase implementations
├── data/               # Repositories and local database
│   ├── local/          # Room entities and DAOs
│   ├── remote/         # API interfaces and DTOs
│   └── repository/     # Repository implementations
├── ml/                 # TFLite inference and preprocessing
│   ├── preprocess/
│   ├── tflite/
│   └── contract/
├── camera/             # CameraX capture logic
└── di/                 # Hilt modules
```

---

## 3. Model Pipeline

### Stage 1: Binary Detection

| Property | Value |
|---|---|
| **Purpose** | OSCC vs Normal tissue |
| **Backbone** | Frozen UNI or CTransPath (primary); EfficientNetB3 (fallback) |
| **Head** | Single linear layer + sigmoid |
| **Input** | 224×224 or 300×300 RGB patch |
| **Output** | Binary label + confidence score |
| **Target** | Sensitivity ≥ 0.95, Specificity ≥ 0.90 |

### Stage 2: Grading

| Property | Value |
|---|---|
| **Purpose** | Grade classification after OSCC detection |
| **Backbone** | Same frozen backbone as Stage 1 |
| **Head** | Multi-class linear layer + softmax |
| **Classes** | `normal` / `osmf` / `wd` / `md` / `pd` |
| **Target** | Accuracy ≥ 0.85 |

### Stage 3: Segmentation (v1.2)

| Property | Value |
|---|---|
| **Purpose** | Tissue component identification |
| **Architecture** | TILSeg-MobileViT |
| **Classes** | Epithelial, stromal, TILs, collagen |
| **Dataset** | NDB-UFES |
| **Target** | mIOU ≥ 0.85 |

---

## 4. Inference Contract

All inference outputs — whether from API, ONNX, or TFLite — must conform to this JSON schema.

### Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "OralPathInferenceResult",
  "type": "object",
  "required": ["case_id", "model_version", "input_quality", "stage1", "disclaimer"],
  "properties": {
    "case_id": {
      "type": "string",
      "description": "Unique identifier for the case"
    },
    "model_version": {
      "type": "string",
      "description": "Semantic version of the model"
    },
    "input_quality": {
      "type": "object",
      "required": ["usable"],
      "properties": {
        "usable": {
          "type": "boolean",
          "description": "Whether the image passes quality checks"
        },
        "blur_score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Sharpness score (1.0 = perfectly sharp)"
        },
        "coverage_score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Tissue coverage score (1.0 = full coverage)"
        },
        "rejection_reason": {
          "type": "string",
          "description": "Reason for rejection if usable is false"
        }
      }
    },
    "stage1": {
      "type": "object",
      "required": ["label", "confidence"],
      "properties": {
        "label": {
          "type": "string",
          "enum": ["normal", "oscc"]
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      }
    },
    "stage2": {
      "type": "object",
      "required": ["label", "confidence"],
      "properties": {
        "label": {
          "type": "string",
          "enum": ["normal", "osmf", "wd", "md", "pd", "null"]
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        }
      }
    },
    "heatmap": {
      "type": "object",
      "properties": {
        "available": {
          "type": "boolean"
        },
        "uri": {
          "type": ["string", "null"],
          "description": "Local or remote path to heatmap image"
        },
        "method": {
          "type": "string",
          "enum": ["gradcam", "attention", "null"]
        }
      }
    },
    "segmentation": {
      "type": "object",
      "description": "v1.2 only",
      "properties": {
        "available": {
          "type": "boolean"
        },
        "regions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "label": {
                "type": "string",
                "enum": ["epithelial", "stromal", "tils", "collagen"]
              },
              "area_fraction": {
                "type": "number"
              }
            }
          }
        }
      }
    },
    "disclaimer": {
      "type": "string",
      "enum": ["research_use_only"]
    }
  }
}
```

### Example Response

```json
{
  "case_id": "OP-2026-0001",
  "model_version": "1.0.0",
  "input_quality": {
    "usable": true,
    "blur_score": 0.92,
    "coverage_score": 0.88
  },
  "stage1": {
    "label": "oscc",
    "confidence": 0.97
  },
  "stage2": {
    "label": "md",
    "confidence": 0.84
  },
  "heatmap": {
    "available": true,
    "uri": "file:///data/data/com.oralpath/heatmaps/op-2026-0001.png",
    "method": "gradcam"
  },
  "segmentation": {
    "available": false,
    "regions": []
  },
  "disclaimer": "research_use_only"
}
```

---

## 5. API Specification

### `POST /predict`

Accepts an image file and returns the inference contract JSON.

**Request**
```
Content-Type: multipart/form-data

image: <image file>
case_id: string (optional, auto-generated if omitted)
```

**Response**
```json
HTTP/1.1 200 OK
Content-Type: application/json

{ /* InferenceResult schema */ }
```

**Error Responses**
```json
HTTP/1.1 400 Bad Request
{ "error": "image_quality_rejected", "reason": "blur_score_too_low" }

HTTP/1.1 422 Unprocessable Entity
{ "error": "inference_failed", "detail": "..." }
```

### `GET /health`

Health check for the inference service.

**Response**
```json
{ "status": "ok", "model_version": "1.0.0", "backbone": "uni" }
```

---

## 6. Data Flow

### Image Capture to Result

1. **Capture** — User captures image via CameraX or selects from gallery
2. **Quality Check** — On-device blur and coverage analysis
3. **Preprocessing** — Resize, normalize, optional Macenko stain normalization
4. **Inference Request** — Image sent to primary API or processed locally via TFLite
5. **Result Parsing** — Response validated against inference contract
6. **UI Render** — Result card, heatmap overlay, plain-language summary
7. **Storage** — Case saved to Room DB with thumbnail

### Case Storage Schema (Room)

| Field | Type | Description |
|---|---|---|
| `case_id` | String (PK) | Unique case identifier |
| `patient_id` | String | User-entered patient ID |
| `date` | Long (timestamp) | Capture timestamp |
| `slide_site` | String | Biopsy site |
| `magnification` | String | Microscope magnification |
| `image_path` | String | Local image file path |
| `heatmap_path` | String? | Local heatmap file path |
| `stage1_label` | String | `normal` or `oscc` |
| `stage1_confidence` | Float | 0.0–1.0 |
| `stage2_label` | String? | `normal`, `osmf`, `wd`, `md`, `pd`, or null |
| `stage2_confidence` | Float? | 0.0–1.0 |
| `disclaimer_shown` | Boolean | Whether disclaimer was acknowledged |

---

## 7. Export Targets

| Format | Purpose | Pipeline |
|---|---|---|
| ONNX | Server-side inference | `torch.onnx.export` |
| TFLite (INT8) | On-device fallback | `tf.lite.TFLiteConverter` with `Optimize.DEFAULT` |

Both exports must produce numerically equivalent results on the same test image within tolerance ±0.01 for confidence scores.

---

*Last updated: June 2026*
