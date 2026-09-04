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

- Model is a CNN with input shape [1, 28, 28, 1] (float 0–1, ink = 1.0, background = 0.0); `KerasTFLite.kt` wraps the flat 784-float array. Android-side preprocessing in `MainActivity.preprocess` must stay MNIST-style: bbox crop → scale to 20px → intensity normalize → center-of-mass centering. `FingerPaintView.exportDrawingBitmap` must export pure white background at native resolution — never draw the view's background color into the export, or the ink threshold and aspect ratio break.
- Maven repos in `settings.gradle.kts` use Aliyun mirrors — do not remove; required for network access in China.
- Gradle daemon JVM toolchain is pinned to version 21 in `gradle/gradle-daemon-jvm.properties` (auto-provisioned on build).
- The venv's pip can break after partial upgrades; repair with `https://bootstrap.pypa.io/pip/3.9/get-pip.py` (venv is Python 3.9).

## Conventions

- Commit messages: Conventional Commits written in Chinese (e.g., `build(gradle): 升级 Gradle 和依赖配置以支持新版本`).
