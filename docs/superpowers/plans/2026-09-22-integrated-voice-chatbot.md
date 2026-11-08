# 整合式語音身分、情緒與中文聊天 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the local Demo so Chinese recordings automatically archive, identify one of four members or an unknown user, classify emotion, transcribe and correct text, then send the combined result to a local Ollama chat model.

**Architecture:** Keep emotion classification, speaker identity, transcription, and chat separate. A shared workspace service owns the local `~/Desktop/專題研究` hierarchy and writes one immutable archive record per input. Identity uses a pre-trained speaker-embedding extractor plus local centroid comparison; emotion keeps the existing XLS-R pipeline; transcription is local Whisper; the Gradio UI combines their outputs and correction controls.

**Tech Stack:** Python 3.12, Gradio 6, PyTorch/Transformers, scikit-learn, FFmpeg, SpeechBrain ECAPA-TDNN, OpenAI Whisper, Ollama local REST API, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-voice-identity-chinese-chatbot-design.md`

## Global Constraints

- Use only local files under `~/Desktop/專題研究`; do not upload recordings or inference requests to a cloud service.
- Keep existing three-label emotion functionality and its data intact until a migration test proves copied data is usable.
- Known speakers are exactly `莊媗丞`, `姜昀廷`, `紀律廷`, `趙科翰`; insufficient-confidence results display `未知使用者`.
- Normalize derived audio to 16 kHz mono WAV and retain the untouched original file.
- Denoise only a derived copy; never overwrite an original recording.
- Fix transcription language to Chinese and store both original and corrected text; calculate character error rate (CER).
- Do not use a China- or Chinese-investment-origin model or tool for training or inference; record exact model source, version and license before downloading weights.
- Preserve macOS light/dark styling and local-only binding at `127.0.0.1`.

## Review Focus

- A malformed, silent, too-short, or undecodable upload must be archived with an error record but never enter any training set. Covered in Task 2.
- Two uploads with the same filename must receive different UUID archive paths and never overwrite each other. Covered in Task 1.
- A confident but near-tied voice match must be `未知使用者`, not arbitrarily one of the four names. Covered in Task 4.
- A correction must update only the targeted label; undo must restore the prior speaker centroid or emotion artifact atomically. Covered in Tasks 4 and 7.
- Chinese text containing punctuation, spaces, or an empty correction must produce a defined CER result without division by zero. Covered in Task 5.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/speech_emotion/workspace.py` | Workspace root, immutable archive copies, JSON metadata and safe paths. |
| `src/speech_emotion/audio.py` | Existing validation plus FFmpeg denoise-and-normalize derivative creation. |
| `src/speech_emotion/speaker_identity.py` | Speaker vector extraction boundary, local centroids, similarity scores, threshold decision and correction history. |
| `src/speech_emotion/transcription.py` | Chinese Whisper adapter, JSON result persistence, correction and CER. |
| `src/speech_emotion/ollama_client.py` | Local-only Ollama availability check and structured Chinese chat request. |
| `src/speech_emotion/integrated_demo.py` | Orchestrates archive → audio processing → three inference modules → combined record. |
| `src/speech_emotion/personal_demo_ui.py` | Extends the existing Gradio page with speaker, transcript, corrections and chat sections. |
| `scripts/run_integrated_demo.py` | Starts the upgraded local page with an automatically selected available port. |
| `tests/test_workspace.py` | Archive and metadata safety tests. |
| `tests/test_speaker_identity.py` | Centroid, threshold, unknown and correction tests using fake embeddings. |
| `tests/test_transcription.py` | Chinese transcription persistence and CER tests using a fake transcriber. |
| `tests/test_ollama_client.py` | Local request payload and unavailable-service tests. |
| `tests/test_integrated_demo.py` | Combined-result and correction workflow tests without real model downloads. |
| `docs/語音身分與聊天系統使用手冊.md` | Student-facing collection, training, correction and demonstration instructions. |

### Task 1: Create the local project workspace and immutable archive

**Files:**
- Create: `src/speech_emotion/workspace.py`
- Create: `tests/test_workspace.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `ProjectWorkspace(root: Path)`, `archive_input(source: Path, source_kind: str) -> ArchivedAudio`, and `write_json(category: str, payload: dict[str, object]) -> Path`.
- `ArchivedAudio` is a dataclass with `record_id: str`, `original_path: Path`, `created_at: str`, and `source_kind: str`.

- [ ] **Step 1: Write failing archive tests**

```python
def test_archive_input_keeps_original_and_assigns_unique_paths(tmp_path, wav_factory):
    source = wav_factory("voice.wav", np.ones(16000, dtype=np.float32))
    workspace = ProjectWorkspace(tmp_path / "專題研究")

    first = workspace.archive_input(source, "upload")
    second = workspace.archive_input(source, "upload")

    assert first.original_path.exists()
    assert second.original_path.exists()
    assert first.original_path != second.original_path
    assert first.original_path.parent.name == "自動留存"
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_workspace.py::test_archive_input_keeps_original_and_assigns_unique_paths -q`

Expected: FAIL because `speech_emotion.workspace` does not exist.

- [ ] **Step 3: Implement the workspace and fixed hierarchy**

```python
DEFAULT_WORKSPACE_ROOT = Path.home() / "Desktop" / "專題研究"

@dataclass(frozen=True)
class ArchivedAudio:
    record_id: str
    original_path: Path
    created_at: str
    source_kind: str

class ProjectWorkspace:
    def archive_input(self, source: Path, source_kind: str) -> ArchivedAudio:
        record_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        target = self.root / "01_原始音檔" / "自動留存" / f"{record_id}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.write_json("05_完整分析紀錄", {"record_id": record_id, "original_path": str(target), "source_kind": source_kind, "created_at": created_at, "status": "archived"})
        return ArchivedAudio(record_id, target, created_at, source_kind)
```

Create all directories named in the approved spec during `ProjectWorkspace.__init__`. Add `~/Desktop/專題研究/` to `.gitignore` only if a test configuration ever locates it under the repository.

- [ ] **Step 4: Add malformed metadata and filename traversal tests, then pass them**

```python
def test_write_json_does_not_allow_category_to_escape_workspace(tmp_path):
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    with pytest.raises(WorkspaceError):
        workspace.write_json("../outside", {"record_id": "x"})
```

Run: `PYTHONPATH=src pytest tests/test_workspace.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/workspace.py tests/test_workspace.py .gitignore
git commit -m "feat: archive integrated voice demo inputs locally"
```

### Task 2: Add derived denoise-and-normalize processing

**Files:**
- Modify: `src/speech_emotion/audio.py`
- Modify: `tests/test_audio.py`

**Interfaces:**
- Consumes `source_path: Path` and a destination path from `ProjectWorkspace`.
- Produces `denoise_and_normalize_audio(source_path: Path, destination_path: Path) -> Path`.

- [ ] **Step 1: Write failing denoise tests**

```python
def test_denoise_and_normalize_creates_valid_16khz_mono_wav(tmp_path, wav_factory):
    source = wav_factory("input.wav", np.ones(32000, dtype=np.float32))
    output = denoise_and_normalize_audio(source, tmp_path / "derived.wav")

    samples, rate = soundfile.read(output)
    assert rate == 16000
    assert samples.ndim == 1

def test_denoise_and_normalize_removes_partial_output_on_invalid_audio(tmp_path):
    with pytest.raises(AudioValidationError):
        denoise_and_normalize_audio(tmp_path / "missing.webm", tmp_path / "derived.wav")
    assert not (tmp_path / "derived.wav").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_audio.py -k denoise -q`

Expected: FAIL because `denoise_and_normalize_audio` does not exist.

- [ ] **Step 3: Implement a derived-only FFmpeg denoise pipeline**

```python
def denoise_and_normalize_audio(source_path: Path, destination_path: Path) -> Path:
    # Build a temporary WAV with FFmpeg's afftdn filter, then validate it.
    _run_ffmpeg(source_path, temporary_wav, audio_filter="afftdn=nf=-25")
    samples, rate = _load_with_librosa(temporary_wav, 16000)
    _validate_audio(samples, rate, 0.5)
    shutil.move(temporary_wav, destination_path)
    return destination_path
```

Extend `_run_ffmpeg` with an optional `audio_filter: str | None` argument; retain the existing normalization behavior for current emotion tests.

- [ ] **Step 4: Run focused and full tests**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest tests/test_audio.py -q && MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest -q -W error::UserWarning -W error::FutureWarning`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/audio.py tests/test_audio.py
git commit -m "feat: create denoised audio derivatives"
```

### Task 3: Build a local speaker enrollment and identity service

**Files:**
- Create: `src/speech_emotion/speaker_identity.py`
- Create: `tests/test_speaker_identity.py`
- Modify: `requirements.txt`
- Modify: `reports/provenance.md`

**Interfaces:**
- Consumes processed WAV paths and `ProjectWorkspace`.
- Produces `enroll_speaker(audio_path: Path, speaker: str, workspace: ProjectWorkspace) -> Path`, `rebuild_speaker_profiles(workspace: ProjectWorkspace, embedder: SpeakerEmbedder) -> dict[str, int]`, and `identify_speaker(audio_path: Path, workspace: ProjectWorkspace, embedder: SpeakerEmbedder, threshold: float, margin: float) -> IdentityResult`.
- `IdentityResult` contains `speaker: str`, `scores: dict[str, float]`, `is_unknown: bool`, and `latency_ms: float`.

- [ ] **Step 1: Write failing centroid and unknown-decision tests with a fake embedder**

```python
class FakeEmbedder:
    def embed(self, path: Path) -> np.ndarray:
        return {"a.wav": np.array([1.0, 0.0]), "b.wav": np.array([0.9, 0.1]), "new.wav": np.array([0.95, 0.05])}[path.name]

def test_identify_returns_known_speaker_with_highest_similarity(workspace, wav_factory):
    # Enroll two speakers using paths whose fake vectors are deterministic.
    result = identify_speaker(Path("new.wav"), workspace, FakeEmbedder(), threshold=0.70, margin=0.05)
    assert result.speaker == "莊媗丞"
    assert result.is_unknown is False

def test_identify_returns_unknown_for_near_tied_or_low_scores(workspace):
    result = identify_speaker(Path("new.wav"), workspace, LowMarginEmbedder(), threshold=0.70, margin=0.05)
    assert result.speaker == "未知使用者"
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_speaker_identity.py -q`

Expected: FAIL because `speaker_identity` does not exist.

- [ ] **Step 3: Implement file-backed enrollment, normalized centroids and correction history**

```python
KNOWN_SPEAKERS = ("莊媗丞", "姜昀廷", "紀律廷", "趙科翰")
UNKNOWN_SPEAKER = "未知使用者"

def identify_speaker(
    audio_path: Path,
    workspace: ProjectWorkspace,
    embedder: SpeakerEmbedder,
    threshold: float,
    margin: float,
) -> IdentityResult:
    vector = _unit(embedder.embed(audio_path))
    scores = {speaker: float(np.dot(vector, _load_centroid(speaker, workspace))) for speaker in KNOWN_SPEAKERS}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] < threshold or ordered[0][1] - ordered[1][1] < margin:
        return IdentityResult(UNKNOWN_SPEAKER, scores, True, latency_ms)
    return IdentityResult(ordered[0][0], scores, False, latency_ms)
```

Store processed enrollments under `01_原始音檔/身分辨識/<speaker>/`; write per-file vectors under `03_語音身分辨識/聲紋向量/` and normalized centroids under `03_語音身分辨識/成員聲紋範本/`. Implement correction history as append-only JSON and only delete a derived enrollment on explicit undo.

- [ ] **Step 4: Add the production adapter behind the tested boundary**

Add `SpeechBrainEmbedder` with `embed(path: Path) -> np.ndarray`. Pin the package version only after verifying Python 3.12 and macOS ARM installation in a disposable virtual environment. Download `speechbrain/spkrec-ecapa-voxceleb` only after adding its source URL, model revision, Apache-2.0 license and non-China-origin review to `reports/provenance.md`.

Run: `PYTHONPATH=src pytest tests/test_speaker_identity.py -q`

Expected: PASS without requiring model download because tests use `FakeEmbedder`.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/speaker_identity.py tests/test_speaker_identity.py requirements.txt reports/provenance.md
git commit -m "feat: add local speaker identity profiles"
```

### Task 4: Add Chinese transcription, correction storage and CER

**Files:**
- Create: `src/speech_emotion/transcription.py`
- Create: `tests/test_transcription.py`
- Modify: `requirements.txt`
- Modify: `reports/provenance.md`

**Interfaces:**
- Produces `transcribe_chinese(audio_path: Path, record_id: str, workspace: ProjectWorkspace, transcriber: ChineseTranscriber) -> TranscriptResult`, `save_transcript_correction(record_id: str, corrected_text: str, workspace: ProjectWorkspace) -> TranscriptResult`, and `character_error_rate(reference: str, hypothesis: str) -> float | None`.
- `TranscriptResult` is a dataclass with `record_id: str`, `original_text: str`, `corrected_text: str | None`, `cer: float | None`, and `latency_ms: float`.

- [ ] **Step 1: Write failing transcription and CER tests**

```python
def test_transcription_persists_original_and_corrected_chinese_text(workspace, wav_factory):
    result = transcribe_chinese(wav_factory("line.wav", np.ones(16000)), "record-1", workspace, FakeTranscriber("今天天氣很好"))
    corrected = save_transcript_correction("record-1", "今天天氣非常好", workspace)
    assert corrected.original_text == "今天天氣很好"
    assert corrected.corrected_text == "今天天氣非常好"

def test_character_error_rate_handles_empty_corrected_text():
    assert character_error_rate("你好", "") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_transcription.py -q`

Expected: FAIL because `transcription` does not exist.

- [ ] **Step 3: Implement JSON persistence and deterministic CER**

```python
def character_error_rate(reference: str, hypothesis: str) -> float | None:
    reference_chars = list(_normalize_text(reference))
    hypothesis_chars = list(_normalize_text(hypothesis))
    if not reference_chars:
        return None
    return levenshtein_distance(reference_chars, hypothesis_chars) / len(reference_chars)
```

Use correction text as reference and original Whisper text as hypothesis when writing `cer`. Save one JSON document per record under `04_中文語音轉文字/原始轉寫結果/`, and append corrections under `人工校正結果/` without deleting original output.

- [ ] **Step 4: Add the production Whisper adapter behind the fake interface**

Add `OpenAIWhisperTranscriber(model_name: str = "base")`; call `model.transcribe(str(audio_path), language="zh", fp16=False)`. Pin the `openai-whisper` version after a CPU/MPS smoke test, and record its exact revision, MIT license and source in `reports/provenance.md`. Do not download model weights during pytest.

Run: `PYTHONPATH=src pytest tests/test_transcription.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/transcription.py tests/test_transcription.py requirements.txt reports/provenance.md
git commit -m "feat: add local Chinese transcript corrections"
```

### Task 5: Build the combined analysis service and automatic correction actions

**Files:**
- Create: `src/speech_emotion/integrated_demo.py`
- Create: `tests/test_integrated_demo.py`
- Modify: `src/speech_emotion/personal_demo.py`

**Interfaces:**
- Produces `analyse_recording(source: Path, source_kind: str, workspace: ProjectWorkspace, services: AnalysisServices) -> CombinedAnalysis`.
- `CombinedAnalysis` contains `record_id`, `archive_path`, `processed_path`, `identity: IdentityResult`, `emotion: dict[str, object]`, `transcript: TranscriptResult`, and `total_latency_ms`.
- Produces `correct_combined_analysis(record_id: str, speaker: str | None, emotion: str | None, transcript: str | None, workspace: ProjectWorkspace, services: AnalysisServices) -> CombinedAnalysis` and `undo_last_combined_correction(workspace: ProjectWorkspace, services: AnalysisServices) -> CombinedAnalysis`.

- [ ] **Step 1: Write failing end-to-end orchestration tests with fake services**

```python
def test_analyse_recording_archives_once_and_returns_three_results(tmp_path, wav_factory):
    result = analyse_recording(wav_factory("input.wav", np.ones(16000)), "upload", ProjectWorkspace(tmp_path / "專題研究"), fake_services())
    assert result.archive_path.exists()
    assert result.identity.speaker == "莊媗丞"
    assert result.emotion["emotion"] == "calm"
    assert result.transcript.original_text == "這是一段測試"

def test_invalid_audio_is_archived_but_not_sent_to_model(tmp_path):
    result = analyse_recording(tmp_path / "broken.webm", "upload", ProjectWorkspace(tmp_path / "專題研究"), fake_services())
    assert result.status == "error"
    assert result.archive_path.exists()
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_integrated_demo.py -q`

Expected: FAIL because `integrated_demo` does not exist.

- [ ] **Step 3: Implement the fixed order and combined JSON record**

```python
def analyse_recording(source, source_kind, workspace, services):
    archive = workspace.archive_input(source, source_kind)
    try:
        processed = denoise_and_normalize_audio(archive.original_path, workspace.processed_audio_path(archive.record_id))
        identity = services.identity.identify(processed)
        emotion = services.emotion.predict(processed)
        transcript = services.transcriber.transcribe(processed, archive.record_id)
        return _persist_success(archive, processed, identity, emotion, transcript, workspace)
    except (AudioValidationError, PersonalDemoError, IdentityError, TranscriptError) as error:
        return _persist_failure(archive, str(error))
```

Re-use `predict_personal_demo` for emotion. Do not train either model from an analysis result unless the user explicitly corrects its label.

- [ ] **Step 4: Add correction and rollback tests, then pass them**

```python
def test_speaker_and_emotion_correction_rebuilds_only_selected_artifacts(workspace, fake_services):
    corrected = correct_combined_analysis(
        "record-1", speaker="紀律廷", emotion=None, transcript="正確文字",
        workspace=workspace, services=fake_services,
    )
    assert corrected.identity.speaker == "紀律廷"
    assert corrected.transcript.corrected_text == "正確文字"
```

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest tests/test_integrated_demo.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/integrated_demo.py src/speech_emotion/personal_demo.py tests/test_integrated_demo.py
git commit -m "feat: combine local voice analysis results"
```

### Task 6: Add a local-only Ollama chat adapter

**Files:**
- Create: `src/speech_emotion/ollama_client.py`
- Create: `tests/test_ollama_client.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces `OllamaClient(base_url: str = "http://127.0.0.1:11434")`, `is_available() -> bool`, and `chat(analysis: CombinedAnalysis, user_message: str, model: str) -> str`.

- [ ] **Step 1: Write failing payload and unavailable-service tests**

```python
def test_chat_sends_identity_emotion_and_corrected_transcript(monkeypatch, combined_analysis):
    captured = {}
    monkeypatch.setattr("speech_emotion.ollama_client.httpx.post", lambda url, json, timeout: captured.update({"url": url, "json": json}) or FakeResponse())
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    reply = client.chat(combined_analysis, "我今天心情不好", "local-chat-model")
    assert "莊媗丞" in captured["json"]["messages"][0]["content"]
    assert "平靜" in captured["json"]["messages"][0]["content"]
    assert reply == "我在這裡聽你說。"

def test_chat_reports_local_service_unavailable_without_network_fallback():
    with pytest.raises(OllamaUnavailableError):
        OllamaClient(base_url="http://127.0.0.1:1").chat(combined_analysis, "測試", "local-chat-model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_ollama_client.py -q`

Expected: FAIL because `ollama_client` does not exist.

- [ ] **Step 3: Implement the restricted local request**

```python
def chat(self, analysis, user_message, model):
    payload = {"model": model, "stream": False, "messages": [{"role": "system", "content": build_system_prompt(analysis)}, {"role": "user", "content": user_message}]}
    response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=30.0)
    response.raise_for_status()
    return str(response.json()["message"]["content"])
```

Reject base URLs whose hostname is not `127.0.0.1` or `localhost`. The system prompt must state that identity/emotion are uncertain model estimates, use Chinese, and respond supportively without claiming to be human.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src pytest tests/test_ollama_client.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/ollama_client.py tests/test_ollama_client.py requirements.txt
git commit -m "feat: connect combined analysis to local Ollama"
```

### Task 7: Extend the Apple-style Gradio page and runner

**Files:**
- Modify: `src/speech_emotion/personal_demo_ui.py`
- Create: `scripts/run_integrated_demo.py`
- Modify: `tests/test_personal_demo.py`
- Modify: `README.md`

**Interfaces:**
- UI consumes `analyse_recording`, `correct_combined_analysis`, `undo_last_combined_correction`, `rebuild_speaker_profiles`, and `OllamaClient`.
- `format_combined_analysis(result: CombinedAnalysis) -> str` produces presentation-safe Traditional Chinese Markdown.

- [ ] **Step 1: Write failing UI formatter and runner tests**

```python
def test_format_combined_analysis_shows_unknown_and_chinese_transcript():
    output = format_combined_analysis(fake_unknown_analysis())
    assert "未知使用者" in output
    assert "中文逐字稿" in output
    assert "情緒分數" in output

def test_integrated_runner_keeps_server_local(monkeypatch):
    # Fake demo captures launch kwargs.
    assert captured["server_name"] == "127.0.0.1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_personal_demo.py -k "combined or integrated_runner" -q`

Expected: FAIL because formatter and runner do not exist.

- [ ] **Step 3: Add grouped UI controls without removing existing emotion controls**

Add an integrated analysis section with a single microphone/upload source, three result panels, and three correction controls:

```python
speaker_correction = gr.Dropdown(choices=[*KNOWN_SPEAKERS, UNKNOWN_SPEAKER], label="正確說話者")
emotion_correction = gr.Dropdown(choices=[("正面", "positive"), ("負面", "negative"), ("平靜", "calm")], label="正確情緒")
transcript_correction = gr.Textbox(label="正確逐字稿", lines=3)
```

Add a “訓練聲紋範本” button, an “分析這段錄音” button, a “儲存更正並更新模型” button, an undo button, and an Ollama chat section. Use existing `APPLE_STYLE_CSS` classes and add no cloud/share option.

- [ ] **Step 4: Run full tests and manually verify both colour schemes**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest -q -W error::UserWarning -W error::FutureWarning`

Then run: `PYTHONPATH=src python scripts/run_integrated_demo.py`

Verify in macOS light and dark appearance: three equal input columns on desktop, one column on narrow view, audio auto-archives before analysis, and UI gives an understandable error if Whisper, speaker profiles or Ollama are unavailable.

- [ ] **Step 5: Commit**

```bash
git add src/speech_emotion/personal_demo_ui.py scripts/run_integrated_demo.py tests/test_personal_demo.py README.md
git commit -m "feat: present integrated local voice chatbot demo"
```

### Task 8: Document data collection, model provenance and demonstration checks

**Files:**
- Create: `docs/語音身分與聊天系統使用手冊.md`
- Modify: `reports/provenance.md`
- Modify: `README.md`

**Interfaces:**
- Documentation consumes the workspace paths, known speakers, model configuration values and runner from Tasks 1–7.

- [ ] **Step 1: Write a failing documentation-presence test**

```python
def test_integrated_manual_documents_all_local_workflows():
    manual = Path("docs/語音身分與聊天系統使用手冊.md").read_text(encoding="utf-8")
    assert "莊媗丞" in manual
    assert "未知使用者" in manual
    assert "CER" in manual
    assert "Ollama" in manual
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_personal_demo.py::test_integrated_manual_documents_all_local_workflows -q`

Expected: FAIL because the manual does not exist.

- [ ] **Step 3: Write the manual and provenance record**

Document: the 20 identity sentence recordings; emotional collection amount; how auto-archive works; how to train profiles and emotion; how to collect non-member test audio; how to correct/undo; how to interpret Accuracy, Macro F1, CER and unknown results; exact local runner commands; stopping the server with Control+C; no-cloud privacy boundary; model sources, revisions and licenses.

- [ ] **Step 4: Run documentation and full regression tests**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src pytest -q -W error::UserWarning -W error::FutureWarning`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/語音身分與聊天系統使用手冊.md reports/provenance.md README.md tests/test_personal_demo.py
git commit -m "docs: add integrated voice chatbot handbook"
```

## Plan Self-Review

- **Spec coverage:** Task 1 implements local storage and hierarchy; Task 2 implements derived-only denoise; Task 3 implements four-speaker/unknown identity; Task 4 implements Chinese transcription, corrections and CER; Task 5 makes combined local records and reversible corrections; Task 6 adds local Ollama; Task 7 supplies the single Apple-style interface; Task 8 supplies collection, source and demonstration documentation.
- **Placeholder scan:** No undecided model, interface, test command or directory has been left as a placeholder. The one deferred value, unknown-user threshold, is intentionally calibrated from collected known and external-friend test audio and is persisted as configuration.
- **Type consistency:** `ProjectWorkspace`, `IdentityResult`, `TranscriptResult`, `CombinedAnalysis`, and the processing paths are introduced before their use by later tasks. Production model adapters remain behind fakeable interfaces so tests do not download weights.
- **Review focus:** Every listed high-risk input has a named owning task and a concrete test.
