# OralPath — Android App

> Non-commercial research project. Mobile-first OSCC diagnostic assistant.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Jetpack Compose |
| Architecture | MVVM + Clean Architecture |
| DI | Hilt |
| Navigation | Jetpack Navigation Component |
| Camera | CameraX |
| Image Processing | OpenCV Android (stain normalization), Glide (display) |
| On-device ML | TFLite |
| Networking | Retrofit 2 + OkHttp3 |
| Serialization | Kotlin Serialization |
| Local Database | Room (SQLite) |
| PDF Generation | Android PdfDocument API |
| Async | Kotlin Coroutines + Flow |
| Testing | JUnit 4, Espresso, MockK |

## Setup

### Prerequisites

- Android Studio Ladybug or newer
- JDK 17
- Android SDK API 26–35

### Build

1. Open `android/` in Android Studio
2. Sync Gradle
3. Run on emulator or device (API 26+)

## Project Structure

```
com.oralpath/
├── ui/                 # Compose screens
│   ├── capture/        # Camera capture + gallery upload
│   ├── result/         # Diagnosis result card + heatmap
│   ├── history/        # Case list
│   └── report/         # PDF preview
├── domain/             # Use cases
├── data/               # Repositories, Room DB
├── ml/                 # TFLite inference, preprocessing
├── camera/             # CameraX wrapper
└── di/                 # Hilt modules
```

## Screens

| Screen | Purpose |
|---|---|
| Capture | Photo via CameraX or gallery upload, quality check |
| Result | Classification, grade, confidence, heatmap overlay |
| History | Local case list with thumbnails |
| Report | PDF preview and export |

## Inference Paths

1. **Primary**: API inference (FastAPI / Hugging Face)
2. **Fallback**: On-device TFLite (bundled or downloaded model)

## Mock Mode

The app can run in mock mode for UI development without a model:
```kotlin
// TODO: add mock inference module
```

---

*Work in progress — scaffold phase.*
