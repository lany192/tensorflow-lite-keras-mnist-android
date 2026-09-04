# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## Project overview

Two-part project for handwritten digit (MNIST) recognition:
- `python/` — Keras training script (`keras_mnist_tflite.py`) that trains a model and converts it to TensorFlow Lite.
- `app/` — single-module Android app (Kotlin, ViewBinding) that runs the model via TFLite to recognize digits drawn in `FingerPaintView`.

## Workflow

- Train model: `cd python && python keras_mnist_tflite.py` → writes `model.tflite` to CWD → copy into `app/src/main/assets/` for the app to load.
- Python deps: `pip install -r python/requirements.txt` (uses Aliyun mirror: `https://mirrors.aliyun.com/pypi/simple/`).
- Android: standard Gradle (`./gradlew assembleDebug`, `./gradlew test`).

## Gotchas

- Model filename inconsistency: `KerasTFLite.kt` loads `assets/keras_mnist_model.tflite`, but the assets folder (and the Python script output) uses `model.tflite`. When adding/updating a model, make the names consistent or the app crashes on load.
- Inference input must match training format: 28×28 float array, normalized to 0–1, inverted (drawn stroke = 1.0, background = 0.0) — see `MainActivity.getPixelData`.
- Maven repos in `settings.gradle.kts` use Aliyun mirrors — do not remove; required for network access in China.
- Gradle daemon JVM toolchain is pinned to version 25 in `gradle/gradle-daemon-jvm.properties` (auto-provisioned on build).

## Conventions

- Commit messages: Conventional Commits written in Chinese (e.g., `build(gradle): 升级 Gradle 和依赖配置以支持新版本`).
