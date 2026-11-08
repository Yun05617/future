# Speech Emotion Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Create a reproducible Python environment and four-class speech-emotion classifier that trains and predicts from completed WAV files.

**Architecture:** A CSV manifest identifies WAV audio, speakers, labels and splits. The program validates audio, uses openSMILE eGeMAPS to extract features, and trains one scikit-learn pipeline containing StandardScaler and SVC. The saved pipeline outputs a label, all four scores and latency.

**Tech Stack:** Python 3.11, venv, openSMILE, soundfile, librosa, NumPy, pandas, scikit-learn, joblib, pytest.

**Spec:** docs/superpowers/specs/2026-09-20-speech-emotion-training-design.md

## Global Constraints

- Use project-local .venv with Python 3.11, not system Python 3.13.
- Classes are exactly neutral, happy, angry and sad.
- Do not commit raw recordings, datasets, trained artifacts, credentials or model weights.
- Split by speaker. A speaker can appear in only one of train, validation and test.
- Log dataset and model source/licence information before competition use.
- Prediction outputs: utterance ID, label, all four scores, status, latency and model version.

## Review Focus

- Missing, unreadable, silent and very short WAV files return a clear error.
- Manifests reject unknown emotions, duplicate paths and a speaker that crosses splits.
- Prediction uses the exact saved feature-column order.
- Training learns the scaler from training rows only.
- Absent classes in a small training set still appear in prediction output with zero scores.

## File Structure

- .gitignore — prevents private/generated files being committed.
- README.md — setup and first-run commands.
- requirements.txt — pinned packages after successful installation.
- data/raw, data/processed, data/manifests — raw audio, generated features and CSV manifests.
- models — saved pipeline artifacts.
- reports/provenance.md — dataset/model licence ledger.
- src/speech_emotion/audio.py — WAV validation.
- src/speech_emotion/manifest.py — CSV validation and speaker split rules.
- src/speech_emotion/features.py — eGeMAPS extraction.
- src/speech_emotion/training.py — pipeline training and persistence.
- src/speech_emotion/prediction.py — saved-model prediction.
- scripts/train.py and scripts/predict.py — command line entry points.
- tests — pytest tests for every unit.

### Task 1: Create the isolated Python 3.11 environment

**Files:**
- Create: .gitignore, README.md, requirements.txt, reports/provenance.md
- Create: data/raw/.gitkeep, data/processed/.gitkeep, data/manifests/.gitkeep, models/.gitkeep
- Create: src/speech_emotion/__init__.py, tests/conftest.py

**Interfaces:**
- Produces: an importable speech_emotion package and activated .venv using Python 3.11.

- [ ] **Step 1: Verify Python 3.11**

Run: python3.11 --version

Expected: Python 3.11.x. If unavailable, install it before continuing.

- [ ] **Step 2: Create the virtual environment**

Run:
~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
~~~

Expected: Python 3.11.x.

- [ ] **Step 3: Create dependency list**

requirements.txt:
~~~text
numpy
pandas
soundfile
librosa
opensmile
scikit-learn
joblib
matplotlib
pytest
~~~

Run:
~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

- [ ] **Step 4: Create privacy exclusions**

.gitignore must include .venv/, __pycache__/, .pytest_cache/, data/raw/* except .gitkeep, data/processed/* except .gitkeep, models/* except .gitkeep, and report image/CSV outputs.

- [ ] **Step 5: Verify the test runner**

tests/conftest.py:
~~~python
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

@pytest.fixture
def wav_factory(tmp_path: Path):
    def create(name: str, samples: np.ndarray, sample_rate: int = 16000) -> Path:
        path = tmp_path / name
        sf.write(path, samples, sample_rate)
        return path
    return create
~~~

Run: PYTHONPATH=src pytest -q

Expected: success with no tests collected.

- [ ] **Step 6: Commit**

Run:
~~~bash
git add .gitignore README.md requirements.txt data models reports src tests
git commit -m "chore: initialize speech emotion environment"
~~~

### Task 2: Validate and load completed WAV files

**Files:**
- Create: src/speech_emotion/audio.py
- Create: tests/test_audio.py

**Interfaces:**
- Produces: load_valid_audio(path: Path, min_duration_s: float = 0.5, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]
- Produces: AudioValidationError(ValueError)

- [ ] **Step 1: Write a failing test**

~~~python
import numpy as np
import pytest
from speech_emotion.audio import AudioValidationError, load_valid_audio

def test_rejects_silent_audio(wav_factory):
    path = wav_factory("silent.wav", np.zeros(16000, dtype=np.float32))
    with pytest.raises(AudioValidationError, match="silent"):
        load_valid_audio(path)
~~~

- [ ] **Step 2: Verify RED**

Run: PYTHONPATH=src pytest tests/test_audio.py -v

Expected: FAIL because speech_emotion.audio does not exist.

- [ ] **Step 3: Implement the minimum**

Use librosa.load(path, sr=16000, mono=True). Reject an unreadable path, samples shorter than 0.5 seconds, and max absolute amplitude below 1e-4. Return a mono float32 array and 16000.

- [ ] **Step 4: Verify GREEN**

Run: PYTHONPATH=src pytest tests/test_audio.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add src/speech_emotion/audio.py tests/test_audio.py && git commit -m "feat: validate wav utterances"

### Task 3: Validate manifests and prevent speaker leakage

**Files:**
- Create: src/speech_emotion/manifest.py
- Create: tests/test_manifest.py

**Interfaces:**
- Consumes: CSV columns path, speaker_id, language, emotion and split.
- Produces: load_manifest(path: Path) -> pandas.DataFrame
- Produces: ManifestValidationError(ValueError)

- [ ] **Step 1: Write failing tests**

~~~python
import pandas as pd
import pytest
from speech_emotion.manifest import ManifestValidationError, load_manifest

def test_rejects_speaker_in_multiple_splits(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame([
        {"path":"a.wav","speaker_id":"S01","language":"zh","emotion":"happy","split":"train"},
        {"path":"b.wav","speaker_id":"S01","language":"zh","emotion":"sad","split":"test"},
    ]).to_csv(path, index=False)
    with pytest.raises(ManifestValidationError, match="multiple splits"):
        load_manifest(path)
~~~

- [ ] **Step 2: Verify RED**

Run: PYTHONPATH=src pytest tests/test_manifest.py -v

Expected: FAIL because speech_emotion.manifest does not exist.

- [ ] **Step 3: Implement the minimum**

Require all five columns; reject duplicate paths; allow only neutral, happy, angry, sad; allow only train, validation, test; reject any speaker whose records have more than one distinct split.

- [ ] **Step 4: Verify GREEN and commit**

Run:
~~~bash
PYTHONPATH=src pytest tests/test_manifest.py -v
git add src/speech_emotion/manifest.py tests/test_manifest.py
git commit -m "feat: validate speaker separated manifests"
~~~

### Task 4: Extract stable eGeMAPS features

**Files:**
- Create: src/speech_emotion/features.py
- Create: tests/test_features.py

**Interfaces:**
- Consumes: mono samples and sample rate from load_valid_audio.
- Produces: extract_egemaps(samples: np.ndarray, sample_rate: int) -> pandas.Series
- Produces: FeatureExtractionError(ValueError)

- [ ] **Step 1: Write the failing test**

~~~python
import numpy as np
from speech_emotion.features import extract_egemaps

def test_returns_finite_named_features():
    samples = np.sin(np.linspace(0, 200, 32000, dtype=np.float32))
    features = extract_egemaps(samples, 16000)
    assert len(features) > 50
    assert features.index.is_unique
    assert np.isfinite(features.to_numpy()).all()
~~~

- [ ] **Step 2: Verify RED**

Run: PYTHONPATH=src pytest tests/test_features.py -v

Expected: FAIL because speech_emotion.features does not exist.

- [ ] **Step 3: Implement**

Create opensmile.Smile with FeatureSet.eGeMAPSv02 and FeatureLevel.Functionals. Call process_signal, require one output row, convert to float Series and reject non-finite values.

- [ ] **Step 4: Verify GREEN and commit**

Run:
~~~bash
PYTHONPATH=src pytest tests/test_features.py -v
git add src/speech_emotion/features.py tests/test_features.py
git commit -m "feat: extract egemaps features"
~~~

### Task 5: Train and persist StandardScaler plus SVC

**Files:**
- Create: src/speech_emotion/training.py
- Create: scripts/train.py
- Create: tests/test_training.py

**Interfaces:**
- Produces: train_pipeline(manifest: pd.DataFrame, audio_root: Path) -> tuple[Pipeline, dict[str, float]]
- Produces: save_model(pipeline: Pipeline, feature_names: list[str], destination: Path) -> None

- [ ] **Step 1: Write a failing test**

~~~python
def test_saved_pipeline_predicts(monkeypatch, tmp_path, manifest):
    monkeypatch.setattr("speech_emotion.training.features_for_path",
                        lambda _: pd.Series({"f1": 0.0, "f2": 1.0}))
    pipeline, metrics = train_pipeline(manifest, tmp_path)
    output = tmp_path / "model.joblib"
    save_model(pipeline, ["f1", "f2"], output)
    assert output.exists()
    assert "validation_macro_f1" in metrics
~~~

The test fixture supplies at least two classes in train and validation.

- [ ] **Step 2: Verify RED**

Run: PYTHONPATH=src pytest tests/test_training.py -v

Expected: FAIL because speech_emotion.training does not exist.

- [ ] **Step 3: Implement**

For every manifest row, call features_for_path. Fit only train rows using:
~~~python
make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
)
~~~
Calculate macro F1 only on validation rows. Persist a joblib dictionary with pipeline, feature_names, labels and model_version.

- [ ] **Step 4: Verify GREEN and commit**

Run:
~~~bash
PYTHONPATH=src pytest tests/test_training.py -v
git add src/speech_emotion/training.py scripts/train.py tests/test_training.py
git commit -m "feat: train speech emotion pipeline"
~~~

### Task 6: Predict one utterance using the saved pipeline

**Files:**
- Create: src/speech_emotion/prediction.py
- Create: scripts/predict.py
- Create: tests/test_prediction.py

**Interfaces:**
- Produces: predict_file(audio_path: Path, model_path: Path, utterance_id: str) -> dict[str, object]

- [ ] **Step 1: Write the failing contract test**

~~~python
def test_prediction_returns_complete_contract(monkeypatch, tmp_path, saved_model):
    monkeypatch.setattr("speech_emotion.prediction.features_for_path",
                        lambda _: {"f1": 0.2})
    result = predict_file(tmp_path / "voice.wav", saved_model, "u-001")
    assert result["utterance_id"] == "u-001"
    assert result["status"] == "ok"
    assert set(result["scores"]) == {"neutral", "happy", "angry", "sad"}
    assert result["latency_ms"] >= 0
~~~

- [ ] **Step 2: Verify RED**

Run: PYTHONPATH=src pytest tests/test_prediction.py -v

Expected: FAIL because speech_emotion.prediction does not exist.

- [ ] **Step 3: Implement**

Load joblib artifact; extract features; reorder by saved feature_names; call predict_proba; set absent class scores to 0.0; record time.perf_counter latency. On failed audio or features, return status error and a user-safe message without a prediction label.

- [ ] **Step 4: Verify GREEN and whole suite**

Run:
~~~bash
PYTHONPATH=src pytest tests/test_prediction.py -v
PYTHONPATH=src pytest -q
~~~

Expected: PASS with no warnings.

- [ ] **Step 5: Commit**

Run: git add src/speech_emotion/prediction.py scripts/predict.py tests/test_prediction.py && git commit -m "feat: predict speech emotion from wav"

### Task 7: Log provenance and run the first experiment

**Files:**
- Modify: reports/provenance.md, README.md
- Create: data/manifests/example_manifest.csv

**Interfaces:**
- Produces: a licence ledger and teammate-reproducible training/prediction commands.

- [ ] **Step 1: Add a failing provenance heading test**

~~~python
def test_provenance_has_required_columns():
    text = Path("reports/provenance.md").read_text(encoding="utf-8")
    assert "Source URL" in text
    assert "Licence" in text
    assert "Decision" in text
~~~

- [ ] **Step 2: Verify RED**

Run: PYTHONPATH=src pytest tests/test_manifest.py::test_provenance_has_required_columns -v

Expected: FAIL until the ledger has headings.

- [ ] **Step 3: Implement the provenance ledger**

Add:
~~~markdown
| Item | Type | Version | Source URL | Licence | Allowed use checked on | Decision |
| --- | --- | --- | --- | --- | --- | --- |
~~~

Add an example manifest using fake speaker IDs and raw/ paths. Add README commands:
~~~bash
PYTHONPATH=src python scripts/train.py --manifest data/manifests/manifest.csv --audio-root data --output models/baseline-1.joblib
PYTHONPATH=src python scripts/predict.py --model models/baseline-1.joblib --audio data/raw/example.wav --utterance-id demo-001
~~~

- [ ] **Step 4: Verify GREEN and commit**

Run:
~~~bash
PYTHONPATH=src pytest -q
git add README.md reports/provenance.md data/manifests/example_manifest.csv tests/test_manifest.py
git commit -m "docs: record speech emotion data provenance"
~~~

## Plan Self-Review

- Spec coverage: Tasks 1-7 cover Python 3.11 setup, audio validation, eGeMAPS features, speaker-safe manifests, training, saved prediction, latency, provenance and reproducibility.
- Placeholder scan: all tasks include exact files, commands and behavior.
- Type consistency: audio produces samples; features produce a Series; training saves feature names; prediction uses saved names.
- Review focus: Task 2 covers audio failures; Task 3 covers labels and speaker leakage; Task 6 covers feature ordering and absent labels.

