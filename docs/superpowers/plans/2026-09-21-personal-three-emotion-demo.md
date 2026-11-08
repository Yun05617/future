# Personal Three-Emotion Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Gradio page that records or uploads personal three-emotion audio, trains a local model, reports same-speaker metrics, and displays probability scores for a new recording.

**Architecture:** A new `personal_demo` module owns only personal-demo data, three-class split/training, model persistence, and prediction. It reuses existing `load_valid_audio()` and `extract_xlsr_embedding()` functions, but stores its own recordings and model artifact. A thin Gradio script wires browser controls to that module without embedding machine-learning logic in the UI.

**Tech Stack:** Python 3.12, Gradio, PyTorch MPS, Transformers XLS-R 300M, pandas, scikit-learn LogisticRegression, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-personal-three-emotion-demo-design.md`

## Global Constraints

- The page runs locally through `python scripts/run_personal_demo.py`; it must not upload recordings or model data to a remote service.
- Labels are exactly `positive`, `negative`, and `calm`; visible interface copy is Traditional Chinese.
- Raw recordings stay in `data/raw/personal_demo/<label>/`, generated state in `data/processed/personal_demo/`, and the model in `models/personal_demo.joblib`; all are ignored by Git.
- Reuse the frozen local XLS-R 300M extractor and MPS when available; do not fine-tune XLS-R.
- Require at least two valid clips per class and label all reported metrics as same-speaker testing only.
- The personal demo is separate from the existing four-class baseline and must not change its result contract.
- Tests must use generated temporary WAV data and fake embeddings; no user recordings or live model downloads in automated tests.

## Review Focus

- A recording uploaded through the browser can be a temporary path: copy it before Gradio removes the temporary file.
- A label with two clips must place exactly one clip in training and one in testing, without duplication.
- Invalid, silent, or short input must not be counted as training data or produce a predicted emotion.
- A model trained with fewer than all three labels must be rejected rather than produce incomplete percentage scores.
- A page opened before model training must show a clear Chinese instruction instead of a stack trace.

---

### Task 1: Add the local Gradio dependency and personal-demo storage helpers

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `src/speech_emotion/personal_demo.py`
- Create: `tests/test_personal_demo.py`

**Interfaces:**
- Consumes: `source_path: Path`, `label: str`, `recordings_root: Path`
- Produces: `add_training_audio(source_path: Path, label: str, recordings_root: Path) -> dict[str, int]`
- Produces: `training_counts(recordings_root: Path) -> dict[str, int]`
- Produces: `PersonalDemoError(ValueError)`

- [ ] **Step 1: Write the failing storage test**

```python
import numpy as np
from speech_emotion.personal_demo import add_training_audio

def test_add_training_audio_copies_valid_clip_and_counts_each_label(tmp_path, wav_factory):
    source = wav_factory("positive.wav", np.ones(16000, dtype=np.float32))
    counts = add_training_audio(source, "positive", tmp_path / "recordings")
    assert counts == {"positive": 1, "negative": 0, "calm": 0}
    assert list((tmp_path / "recordings" / "positive").glob("*.wav"))
```

- [ ] **Step 2: Run the test to verify RED**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_personal_demo.py::test_add_training_audio_copies_valid_clip_and_counts_each_label -v`

Expected: FAIL because `speech_emotion.personal_demo` does not exist.

- [ ] **Step 3: Implement minimal validated storage**

```python
PERSONAL_LABELS = ("positive", "negative", "calm")

def training_counts(recordings_root: Path) -> dict[str, int]:
    return {label: len(list((recordings_root / label).glob("*.wav"))) for label in PERSONAL_LABELS}

def add_training_audio(source_path: Path, label: str, recordings_root: Path) -> dict[str, int]:
    if label not in PERSONAL_LABELS:
        raise PersonalDemoError("unsupported personal-demo label")
    load_valid_audio(source_path)
    destination = recordings_root / label / f"{uuid4().hex}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return training_counts(recordings_root)
```

Wrap `AudioValidationError` as `PersonalDemoError` with a Chinese-safe message. Add `gradio==6.28.0` to `requirements.txt`. Add `data/processed/personal_demo/` and generated personal-demo report files to `.gitignore`.

- [ ] **Step 4: Verify GREEN and full suite**

Run: `source .venv/bin/activate && MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest -q -W error::FutureWarning`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore src/speech_emotion/personal_demo.py tests/test_personal_demo.py
git commit -m "feat: store personal demo recordings"
```

### Task 2: Train and score a three-class personal model

**Files:**
- Modify: `src/speech_emotion/personal_demo.py`
- Modify: `tests/test_personal_demo.py`

**Interfaces:**
- Consumes: `recordings_root: Path`, `model_path: Path`
- Produces: `train_personal_demo(recordings_root: Path, model_path: Path) -> dict[str, object]`
- Persists: a joblib dictionary containing `pipeline`, `feature_names`, `labels`, `model_version`, and `split_paths`

- [ ] **Step 1: Write the failing split-and-training test**

```python
def test_trains_with_two_clips_per_label_and_reports_same_speaker_metrics(monkeypatch, tmp_path):
    root = tmp_path / "recordings"
    for label in ("positive", "negative", "calm"):
        for index in range(2):
            path = root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda path: pd.Series({"f1": float(path.parent.name == "positive"), "f2": float(path.parent.name == "negative")}),
    )
    result = train_personal_demo(root, tmp_path / "personal.joblib")
    assert result["same_speaker_test_count"] == 3
    assert isinstance(result["same_speaker_accuracy"], float)
    assert (tmp_path / "personal.joblib").exists()
```

- [ ] **Step 2: Run the test to verify RED**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_personal_demo.py::test_trains_with_two_clips_per_label_and_reports_same_speaker_metrics -v`

Expected: FAIL because `train_personal_demo` does not exist.

- [ ] **Step 3: Implement deterministic per-label holdout and LogisticRegression**

```python
def _split_paths(paths_by_label: dict[str, list[Path]]) -> tuple[list[Path], list[str], list[Path], list[str]]:
    train_paths, train_labels, test_paths, test_labels = [], [], [], []
    for label, paths in paths_by_label.items():
        ordered = sorted(paths)
        holdout_count = max(1, round(len(ordered) / 3))
        holdout_count = min(holdout_count, len(ordered) - 1)
        test_group, train_group = ordered[:holdout_count], ordered[holdout_count:]
        train_paths.extend(train_group)
        train_labels.extend([label] * len(train_group))
        test_paths.extend(test_group)
        test_labels.extend([label] * len(test_group))
    return train_paths, train_labels, test_paths, test_labels
```

Require at least two WAV paths for every label before splitting. Build feature tables through `features_for_path`, fit `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))`, calculate `accuracy_score` and macro `f1_score`, and save the artifact with `model_version = "personal-three-emotion-v1"`. Return counts and metrics in a dictionary whose metric keys are `same_speaker_accuracy`, `same_speaker_macro_f1`, `same_speaker_train_count`, and `same_speaker_test_count`.

- [ ] **Step 4: Add and run missing-class review-focus test**

```python
def test_rejects_training_when_a_label_has_fewer_than_two_clips(tmp_path):
    root = tmp_path / "recordings"
    for label, count in {"positive": 2, "negative": 2, "calm": 1}.items():
        for index in range(count):
            path = root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    with pytest.raises(PersonalDemoError, match="至少 2 段"):
        train_personal_demo(root, tmp_path / "personal.joblib")
```

Run: `source .venv/bin/activate && MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest tests/test_personal_demo.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/personal_demo.py tests/test_personal_demo.py
git commit -m "feat: train personal three-emotion demo"
```

### Task 3: Predict dynamic three-class probabilities from the saved demo model

**Files:**
- Modify: `src/speech_emotion/personal_demo.py`
- Modify: `tests/test_personal_demo.py`

**Interfaces:**
- Consumes: `audio_path: Path`, `model_path: Path`
- Produces: `predict_personal_demo(audio_path: Path, model_path: Path) -> dict[str, object]`

- [ ] **Step 1: Write the failing prediction contract test**

```python
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def personal_model(tmp_path):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], ["positive", "negative", "calm"])
    model_path = tmp_path / "personal.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": ["f1", "f2"],
        "labels": ["positive", "negative", "calm"],
        "model_version": "personal-three-emotion-v1",
    }, model_path)
    return model_path

def test_predict_personal_demo_returns_all_three_probabilities(monkeypatch, tmp_path, personal_model):
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda _: pd.Series({"f1": 0.2, "f2": 0.3}),
    )
    result = predict_personal_demo(tmp_path / "clip.wav", personal_model)
    assert result["status"] == "ok"
    assert set(result["scores"]) == {"positive", "negative", "calm"}
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6
    assert result["emotion"] in result["scores"]
```

- [ ] **Step 2: Run the test to verify RED**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_personal_demo.py::test_predict_personal_demo_returns_all_three_probabilities -v`

Expected: FAIL because `predict_personal_demo` does not exist.

- [ ] **Step 3: Implement safe dynamic prediction**

Load the demo artifact, extract features, order them using the saved `feature_names`, call `predict_proba`, and construct all three label keys in `PERSONAL_LABELS`, defaulting any absent class to `0.0`. Measure `perf_counter()` latency. On any model or audio failure return only `status`, a Chinese `message`, and `latency_ms`; omit `emotion` and `scores`.

- [ ] **Step 4: Add and run unavailable-model review-focus test**

```python
def test_predict_personal_demo_returns_safe_error_without_model(tmp_path):
    result = predict_personal_demo(tmp_path / "clip.wav", tmp_path / "missing.joblib")
    assert result["status"] == "error"
    assert "emotion" not in result
```

Run: `source .venv/bin/activate && MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest tests/test_personal_demo.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/personal_demo.py tests/test_personal_demo.py
git commit -m "feat: predict personal three-emotion scores"
```

### Task 4: Build the local Gradio page and Chinese results formatting

**Files:**
- Create: `src/speech_emotion/personal_demo_ui.py`
- Create: `scripts/run_personal_demo.py`
- Modify: `tests/test_personal_demo.py`

**Interfaces:**
- Produces: `build_demo() -> gr.Blocks`
- Produces: `format_analysis(result: dict[str, object]) -> str`
- Produces: `format_training(result: dict[str, object]) -> str`

- [ ] **Step 1: Write the failing Chinese display test**

```python
from speech_emotion.personal_demo_ui import format_analysis

def test_format_analysis_displays_label_percentages_and_latency():
    output = format_analysis({
        "status": "ok", "emotion": "positive",
        "scores": {"positive": 0.72, "negative": 0.18, "calm": 0.10},
        "latency_ms": 850.4, "model_version": "personal-three-emotion-v1",
    })
    assert "最終判斷：正面" in output
    assert "正面：72.0%" in output
    assert "分析時間：850.4 ms" in output
```

- [ ] **Step 2: Run the test to verify RED**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_personal_demo.py::test_format_analysis_displays_label_percentages_and_latency -v`

Expected: FAIL because `speech_emotion.personal_demo_ui` does not exist.

- [ ] **Step 3: Implement a local-only Blocks page**

Create a `gr.Blocks` page with a Traditional Chinese title and three areas:

1. **加入訓練錄音**: `gr.Dropdown` for the three labels, `gr.Audio(sources=["upload", "microphone"], type="filepath")`, an Add button, and a Markdown status that shows per-label counts.
2. **訓練個人 Demo**: a Train button and Markdown output from `format_training()`, explicitly containing `同一位說話者測試` before Accuracy and Macro F1.
3. **立即分析**: a separate `gr.Audio` with the same sources, an Analyse button, and Markdown output from `format_analysis()`.

Use fixed paths relative to the repository root: `data/raw/personal_demo` and `models/personal_demo.joblib`. The add callback calls `add_training_audio`; the training callback calls `train_personal_demo`; the analysis callback calls `predict_personal_demo`. Every callback catches `PersonalDemoError` and returns a Chinese instruction rather than raising.

`scripts/run_personal_demo.py` imports `build_demo`, builds once, and launches with `server_name="127.0.0.1"` and `inbrowser=True`; do not enable public sharing.

- [ ] **Step 4: Verify GREEN, launch check, and full suite**

Run:

```bash
source .venv/bin/activate
MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest -q -W error::FutureWarning
PYTHONPATH=src python scripts/run_personal_demo.py --help
```

Expected: all tests PASS; the runner provides a documented local launch path or exits cleanly for `--help`.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/personal_demo_ui.py scripts/run_personal_demo.py tests/test_personal_demo.py
git commit -m "feat: add local personal emotion demo"
```

### Task 5: Document and manually verify the personal demo

**Files:**
- Modify: `docs/語音情緒辨識使用手冊.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the local runner from Task 4
- Produces: beginner-readable personal-demo launch and use instructions

- [ ] **Step 1: Write a failing documentation-presence test**

```python
def test_manual_documents_personal_demo_runner():
    manual = Path("docs/語音情緒辨識使用手冊.md").read_text(encoding="utf-8")
    assert "run_personal_demo.py" in manual
    assert "同一位說話者測試" in manual
```

- [ ] **Step 2: Run the test to verify RED**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_personal_demo.py::test_manual_documents_personal_demo_runner -v`

Expected: FAIL until the usage manual adds the demo section.

- [ ] **Step 3: Document exact setup and first run**

Add a **個人三情緒網頁 Demo** section with:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_personal_demo.py
```

Explain that the user adds at least two clips for each label, presses Train, then uses the separate analysis recorder/uploader. State that the reported Accuracy and Macro F1 test the same speaker only, and recommend at least 10 clips per class before comparing results.

- [ ] **Step 4: Verify documentation, suite, and manual workflow**

Run:

```bash
source .venv/bin/activate
MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest -q -W error::FutureWarning
```

Then manually launch the page, add six non-sensitive local clips, train once, and analyse one new local clip. Confirm that the page reports Chinese percentages and a final label. Do not add these recordings or artifacts to Git.

Expected: all automated tests PASS; manual page accepts both microphone and upload controls.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/語音情緒辨識使用手冊.md tests/test_personal_demo.py
git commit -m "docs: explain personal emotion demo"
```

## Plan Self-Review

- Spec coverage: Tasks 1-4 implement local recording/upload, local storage, XLS-R feature reuse, three labels, deterministic same-speaker testing, probabilities, final labels, latency, errors, and a local-only Gradio page. Task 5 documents launch and manual verification.
- Placeholder scan: no deferred implementation markers or vague test instructions remain.
- Type consistency: Task 1 defines label storage used by Task 2; Task 2's joblib artifact is consumed by Task 3; Task 3's result dictionary is consumed by Task 4; Task 4's runner is documented by Task 5.
- Review focus coverage: temporary upload copying is tested in Task 1; two-per-label split in Task 2; invalid audio is inherited from the tested audio validator and wrapped in Task 1; incomplete class data is tested in Task 2; missing model is tested in Task 3.
