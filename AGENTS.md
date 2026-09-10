# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## Project overview

Two-part project for handwritten digit (MNIST) recognition:
- `python/` — Keras training script (`keras_mnist_tflite.py`) that trains a model and converts it to TensorFlow Lite.
- `app/` — single-module Android app (Kotlin, ViewBinding) that runs the model via TFLite to recognize a multi-digit number drawn in `FingerPaintView`. The canvas is segmented into individual digits, each digit is classified separately, and the results are concatenated left-to-right.

The model still classifies a single character — multi-digit support lives entirely on the Android side.

## Workflow

- Train model: `cd python && python keras_mnist_tflite.py` → writes `model.tflite` to CWD → copy into `app/src/main/assets/` for the app to load.
- Python deps: `pip install -r python/requirements.txt` (uses Aliyun mirror: `https://mirrors.aliyun.com/pypi/simple/`).
- Android: standard Gradle (`./gradlew assembleDebug`, `./gradlew test`). `./gradlew test` exercises the segmentation algorithm — run it before and after touching `InkSegmenter`. APK output: `app/build/outputs/apk/debug/app-debug.apk`.

## Gotchas

- Model is a CNN with input shape [1, 28, 28, 1] (float 0–1, ink = 1.0, background = 0.0); `KerasTFLite.kt` wraps the flat 784-float array. Android-side preprocessing in `MnistPreprocessor.preprocessRegion` must stay MNIST-style: bbox crop → scale to 20px → intensity normalize → center-of-mass centering. `FingerPaintView.exportDrawingBitmap` must export pure white background at native resolution — never draw the view's background color into the export, or the ink threshold and aspect ratio break.
- Multi-digit support is split across three files with a hard boundary:
  - `InkSegmenter.kt` — pure Kotlin, **no `android.*` imports anywhere in the file**. Connected-component segmentation of the ink mask. This is the only part of the pipeline that can be unit-tested on the JVM (`app/src/test/kotlin/.../InkSegmenterTest.kt`), so keep all geometry here and all Bitmap work out.
  - `MnistPreprocessor.kt` — Android. Values here must stay byte-for-byte equivalent to the historical single-digit `preprocess()`; the only change is that the crop box now comes from the segmenter instead of an internal bbox scan. Do not "optimize" the rescale steps or replace `Bitmap.createScaledBitmap` — its private bilinear implementation is what the model was tuned against.
  - `MainActivity.kt` — orchestration only.
- ink/background thresholding (32) lives **only** in `SegmentConfig.inkThreshold`. `MnistPreprocessor` does no thresholding at all — it crops the box the segmenter hands it. If a second threshold appears anywhere, the crop boxes and the normalization will disagree about what counts as ink.
- For a lone segment on the canvas, the neck-split guards reduce to a pure aspect-ratio test (`W < splitAspectFactor * H`, 1.35 by default), so normal 0–9 handwriting is never split. That is a ratio property, not a structural guarantee — lowering `splitAspectFactor`, or a digit written flat enough to exceed 1.35, still triggers a split. `InkSegmenterTest.singleDigit_*` pins this.
- `FingerPaintView.strokeWidth` is **not** a cosmetic choice — measured accuracy is dominated by the ratio `strokeWidth / drawn digit height`, not by the segmentation algorithm or the model: the safe band is ≤0.19, degradation starts at 0.20–0.25, and ≥0.30 collapses. The current 32f keeps that ratio safe for digits roughly 170–800px tall. Raising it back toward 64f makes 4+ digit strings unusable (measured whole-string accuracy at 4 digits drops from 87% to 7%). Re-measure before changing it.
- Maven repos in `settings.gradle.kts` use Aliyun mirrors — do not remove; required for network access in China.
- Gradle daemon JVM toolchain is pinned to version 17 in `gradle/gradle-daemon-jvm.properties` (already installed locally; auto-provisioned if missing).
- The venv's pip can break after partial upgrades; repair with `https://bootstrap.pypa.io/pip/3.9/get-pip.py` (venv is Python 3.9).

## Conventions

- Commit messages: Conventional Commits written in Chinese (e.g., `build(gradle): 升级 Gradle 和依赖配置以支持新版本`).
