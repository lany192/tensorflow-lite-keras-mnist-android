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
- Math practice (`MathPracticeActivity` + `MathProblem.kt` + `MathProblemGenerator.kt`) is built **on top of** the recognition pipeline and does not modify it. `MathProblemGenerator.kt` shares `InkSegmenter.kt`'s hard boundary: pure Kotlin, **no `android.*` import anywhere in the file**, covered by JVM unit tests (`MathProblemGeneratorTest.kt`). Adding an Android import there silently breaks `./gradlew test`.
- **The answer of every generated problem is an integer in 1..9999 because of the recognition pipeline, not by product preference.** The model reads only the digits 0-9, so an answer carrying a decimal point or a fraction bar cannot be judged; the 4-digit ceiling comes from the same stroke-width/digit-height ratio that caps `MAX_DIGITS`. That single constraint forces three mathematical compromises in grades 5-6, all pinned by assertions in `MathProblemGeneratorTest`: decimal subtraction can never borrow (both operands must share the same decimal place, so `5.2 - 1.4` is unconstructible), fraction subtraction must use an improper minuend, and fraction addition can only ever sum to 1. Read those tests before changing any generation rule.
- Answer 0 is deliberately excluded — a lone "0" on the canvas is a closed loop that is hard to tell apart from "nothing written" on some segmentation paths.
- `FingerPaintView.inputEnabled` is the only way to freeze the canvas. `setEnabled(false)` does not work here: the view overrides `onTouchEvent` unconditionally and always returns true.
- `FingerPaintView.clear()` must stay null-safe. `drawingBitmap` is created in `onSizeChanged`, so clearing before the first layout — which `MathPracticeActivity.onCreate` does to present a fresh problem — hits a null bitmap and crashed the app. Clearing an unlaid-out canvas is legitimate; it just has no bitmap to recreate yet.
- Anything placed around the practice canvas that toggles between `GONE` and `VISIBLE` will resize it: the canvas takes `weight=1`, and `onSizeChanged` rebuilds `drawingBitmap`, silently wiping whatever the student just wrote — they see it as "submitting ate my answer". The check hint uses `INVISIBLE` to keep its row reserved. Give any new sibling control the same treatment, or reserve a fixed height.
- Judging an answer must compare **numerically** (`recognizedText.toIntOrNull()`), never as strings — a student writing "068" means 68, and leading zeros are common in handwriting.
- Corollary to the point above: **a control's XML default visibility must equal what `render()` produces for the initial state.** `render()` first runs from `onCreate` and again on every `onStart` (`repeatOnLifecycle`); if the defaults disagree, that first render flips a sibling's visibility, shrinks the canvas and eats the student's handwriting. `textCheckHint` being `invisible` rather than `gone` is this rule in practice.
- `configChanges` was deliberately removed from `MathPracticeActivity`: the ViewModel now holds the state, so rotation/multi-window/foldable/font-scale all rebuild the Activity. Problems, progress, grade, attempts and the current phase survive; canvas strokes and the TFLite interpreter do not (the former is `onSizeChanged` rebuilding the bitmap, the latter is `KerasTFLite` re-copying the model from assets). Note `configChanges` never protected the canvas anyway — it only ever protected the state machine, which the ViewModel now owns.
- The grade `Spinner` is a **second source of truth** for the selected grade (it restores its own selection). `render()` only writes to it when the selection disagrees with `state.grade`, and `MathPracticeViewModel` ignores `SelectGrade` for the grade it already has — that gate is what stops a render→dispatch→render loop, and it is also what absorbs the Spinner's initial `position = 0` callback. Do not remove it.
- ink/background thresholding (32) lives **only** in `SegmentConfig.inkThreshold`. `MnistPreprocessor` does no thresholding at all — it crops the box the segmenter hands it. If a second threshold appears anywhere, the crop boxes and the normalization will disagree about what counts as ink.
- For a lone segment on the canvas, the neck-split guards reduce to a pure aspect-ratio test (`W < splitAspectFactor * H`, 1.35 by default), so normal 0–9 handwriting is never split. That is a ratio property, not a structural guarantee — lowering `splitAspectFactor`, or a digit written flat enough to exceed 1.35, still triggers a split. `InkSegmenterTest.singleDigit_*` pins this.
- `FingerPaintView.strokeWidth` is **not** a cosmetic choice — measured accuracy is dominated by the ratio `strokeWidth / drawn digit height`, not by the segmentation algorithm or the model: the safe band is ≤0.19, degradation starts at 0.20–0.25, and ≥0.30 collapses. The current 32f keeps that ratio safe for digits roughly 170–800px tall. Raising it back toward 64f makes 4+ digit strings unusable (measured whole-string accuracy at 4 digits drops from 87% to 7%). Re-measure before changing it.
- Maven repos in `settings.gradle.kts` use Aliyun mirrors — do not remove; required for network access in China.
- Gradle daemon JVM toolchain is pinned to version 17 in `gradle/gradle-daemon-jvm.properties` (already installed locally; auto-provisioned if missing).
- The venv's pip can break after partial upgrades; repair with `https://bootstrap.pypa.io/pip/3.9/get-pip.py` (venv is Python 3.9).

## MVI conventions

Both screens (`MainActivity`, `MathPracticeActivity`) are MVI: each has a `XxxContract.kt` (State / Intent / Effect) plus a `XxxViewModel.kt`. An Activity only does three things — translate input into an Intent, subscribe to State and render, execute one-shot Effects.

- **`*Contract.kt` and `*ViewModel.kt` must not `import android.`** (`androidx.*` is fine — `androidx.lifecycle.ViewModel` is a plain JVM class). Enforced by `PureKotlinBoundaryTest`. A ViewModel that touches `Bitmap` or `Context` drops out of JVM unit testing entirely — there is no Robolectric here.
- **ViewModels stay synchronous on purpose.** `MutableStateFlow.value =` and `Channel.trySend` never suspend, so `dispatch()` returns with the state already in place and tests assert `state.value` directly — no `kotlinx-coroutines-test`, no `TestDispatcher`. The harder reason: `Dispatchers.Main` does not exist in JVM tests (`coroutines-android` is an Android artifact), so the moment a `viewModelScope.launch` appears the tests fail with `Module with the Main dispatcher had failed to initialize` rather than failing to compile. If you need async, read `MathPracticeViewModel`'s class comment first — it also explains how to lift `reduce` into a pure function when that day comes.
- **All state transitions happen inside the single `private fun reduce(intent)`.** Helper methods may only return a `Transition`; they must not touch `_state.value` directly.
- **Effect carries one-shot events only. Anything a freshly-created Activity must reproduce belongs in State.** The canvas freeze is the worked example: it is a pure function of `phase is Answering`, so `render()` derives it. Making it an Effect would mean that after rotation the new Activity's canvas is writable while the state still says "confirming" — the student thinks the canvas is locked, scribbles freely, then hits Confirm and gets judged on the earlier number.
- **`render()` must be idempotent** (it replays on every `onStart`) and **must never call `dispatch()`**.
- Never put `Bitmap` / `IntArray` / anything identity-equals into State — `StateFlow` conflates on `equals`, which turns into missed or spurious renders.
- Recognition stays in the Activity deliberately: `MnistRecognizer` takes a `Bitmap`. The Activity only turns the canvas into a pure-data `DigitInput` (`CanvasEmpty` / `NotRecognized` / `Digits`); deciding what to *do* about an empty canvas, an unrecognized stroke, or too many digits is ViewModel policy — those are the only branches in the flow, and they have to be testable on the JVM.
- The ViewModels get a no-arg constructor by giving every primary-constructor parameter a default. **Delete a default and `by viewModels()` throws `Cannot create an instance of class` at runtime with no compile-time warning.**

## Conventions

- Commit messages: Conventional Commits written in Chinese (e.g., `build(gradle): 升级 Gradle 和依赖配置以支持新版本`).
