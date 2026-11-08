# Speech Emotion Training Environment Design

## Goal

Build a reproducible Python environment for a Chinese-first speech emotion classifier. The first model classifies one completed utterance as `neutral`, `happy`, `angry`, or `sad`, and returns a label plus class scores. It will later provide one input to a multimodal chatbot alongside a separate Google Teachable Machine facial-expression module.

## Constraints

- Development computer: MacBook Air M3, 16 GB memory.
- Primary language: Mandarin Chinese.
- Interaction: record one utterance, then analyse it.
- Target: begin a useful text response within five seconds after the user stops recording. This is a whole-system target; the speech-emotion module must measure its own latency.
- Team: four members. Two members work on voice; two work on facial-expression recognition.
- Competition: do not use a core open-source model from China or with Chinese investment. Model and data provenance must be recorded before use.
- Initial delivery: a three-minute demonstration video and competition materials due before the 2026-10-05 deadline.

## Selected First Model

The first trainable baseline uses frozen Meta XLS-R 300M multilingual speech embeddings and a scikit-learn support-vector classifier (SVC):

`WAV audio -> audio validation and resampling -> XLS-R 300M embedding -> StandardScaler -> SVC -> label and scores`

XLS-R is an Apache-2.0 multilingual model from Meta whose official model card describes 128-language pretraining. It is kept frozen on the MacBook; only the small StandardScaler plus SVC classifier is trained from the team's labelled recordings. The StandardScaler and SVC are saved together in one scikit-learn pipeline, ensuring prediction uses the identical scaling learnt during training.

The model is a baseline, not a claim that acoustic features alone reveal a user's true internal state. Output is presented as the detected emotional expression in the audio. The application must allow an `uncertain` result when a future validation rule finds insufficient confidence.

## Data Plan

The preferred training data is a legally obtained Mandarin emotional-speech corpus with speaker identifiers and labels for neutral, happy, angry, and sad. ESD is a candidate only after its research-use license and competition suitability are confirmed.

Self-recorded audio is used in two roles:

1. Two voice-team members provide development recordings to find integration and microphone issues.
2. Four to six volunteer friends provide one-time usability and held-out speaker tests after consent. These recordings must not be reused for training or tuning once designated as held-out tests.

Each source audio file has a row in a manifest with: file path, source dataset, speaker ID, language, emotion label, sentence ID when known, recording session, split, and consent/licence reference. Raw audio remains unchanged. Derived feature files are reproducible from the manifest.

All dataset sources, licenses, model versions, and code package versions are recorded in `reports/provenance.md` before they are used in a competition build.

## Dataset Splits and Evaluation

Splits occur by speaker, not by individual recording. A speaker cannot appear in more than one of training, validation, and test. Any clipped or augmented form of an original clip stays in the original speaker's split.

The report will include macro F1, per-class recall, a confusion matrix, the number of speakers and recordings per split, rejected or uncertain-input rate, and feature-extraction plus prediction latency. Chinese results are reported separately from any later language extension.

## Project Structure

```
speech-emotion/
  data/
    raw/             # ignored: original audio and external data
    manifests/       # CSV metadata only
    processed/       # ignored: generated features
  models/            # ignored: trained artifacts
  reports/           # metrics, provenance, figures
  src/               # reusable data, features, training, prediction code
  tests/             # automated behaviour checks
  scripts/           # explicit command-line entry points
```

The directory is created inside the shared project workspace. No external data, credentials, model weights, or recordings are committed to version control.

## Environment

Use Python 3.12 in a project-local `.venv`. Dependencies are pinned in `requirements.txt` after successful installation. Initial dependencies are Python packages for numerical data, audio loading, XLS-R inference through PyTorch and Transformers, scikit-learn training, charts, test execution, and serialized model artifacts. The first version uses Apple Silicon MPS when available and does not require CUDA, TensorFlow, or cloud GPUs.

## First Vertical Slice

1. Set up Python 3.11 and a local virtual environment.
2. Add a manifest schema and audio-validation function.
3. Given a local WAV file, extract an XLS-R embedding vector.
4. Train a scaler-plus-SVC pipeline from a small manifest and persist it.
5. Load the saved pipeline to predict one WAV file and output label, scores, and latency.

Each functional unit is tested before its implementation. Tests use generated temporary WAV data or small non-sensitive fixtures; no participant recording is committed.

## Error Handling

- Reject missing, unreadable, empty, excessively short, or unsupported audio with a clear message.
- Reject training manifests missing required fields, labels outside the four agreed classes, duplicated paths, or speaker overlap across splits.
- Do not produce an emotional label when feature extraction fails.
- Preserve the raw output scores for later calibration and debugging.

## Later Integration Contract

The speech module returns a structured result containing:

```json
{
  "utterance_id": "unique-id",
  "speech_emotion": "happy",
  "scores": {"neutral": 0.1, "happy": 0.7, "angry": 0.1, "sad": 0.1},
  "status": "ok",
  "latency_ms": 320,
  "model_version": "baseline-1"
}
```

The chatbot does not treat this as a statement of the user's true emotion. It uses the result as one optional communication-style signal. A future integration step will combine it with facial-expression output without averaging incomparable score scales.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Mandarin corpus cannot be licensed in time | Train the pipeline on permitted self-recorded data for a prototype, report the two-speaker limitation, and use the flow for the video; seek additional volunteer testing. |
| Dataset model/source restriction | Maintain provenance records and exclude sources that violate the competition rule; ask the organiser where necessary. |
| Model memorizes speakers or recording room | Split by speaker and test on held-out people. |
| Baseline misses nuanced emotion | Report errors; compare an allowed second approach only after the baseline works. |
| Latency exceeds five seconds once modules combine | Measure module and end-to-end latency separately; load models before recording and reduce input length only after testing. |

## Definition of Done for the Baseline

- A new WAV can be validated, featurized, and classified through a saved model.
- The training process is repeatable from a manifest and fixed requirements.
- Speaker-separated validation metrics and latency are generated.
- Dataset and model provenance is recorded.
- No sensitive recordings, model downloads, or secrets enter version control.
