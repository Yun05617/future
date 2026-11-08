# Personal Three-Emotion Demo Design

## Goal

Provide a local web demo for one person's early speech-emotion experiment. The user can record from the browser microphone or upload an audio file, label it as `positive`, `negative`, or `calm`, train a local three-class classifier, and analyse a new recording. The analysis page shows each class percentage, the highest-scoring class, and inference latency.

## User and Scope

The immediate user is the student operating the project on a MacBook Air M3. The purpose is to verify the full audio-to-emotion flow before collecting a larger speaker-separated dataset for the competition. The demo is intentionally a personal experiment, not evidence that the classifier generalises to other people.

The demo does not replace the existing four-class, speaker-separated baseline. It uses a separate model artifact and separate local data directory.

## User Flow

1. Open a local Gradio web page by running `python scripts/run_personal_demo.py` in the activated virtual environment.
2. In **Add training recording**, select one of `positive`, `negative`, or `calm`; record through the microphone or upload an audio file; press Add.
3. Repeat until each class has at least two valid recordings.
4. Press **Train personal demo**. The application preserves a stratified held-out subset, trains a small three-class classifier on the remaining recordings, and shows same-speaker Accuracy and Macro F1.
5. In **Analyse new recording**, record or upload a new audio file and press Analyse.
6. Read a result containing a final label, class scores in percent, and latency.

## Data and Storage

- Original demo recordings are stored only under `data/raw/personal_demo/<label>/` and remain ignored by Git.
- The generated demo manifest and split metadata are stored under `data/processed/personal_demo/` and remain ignored by Git.
- The trained demo model is stored as `models/personal_demo.joblib`, also ignored by Git.
- The page must not upload recordings or model data to any remote service.

## Model Design

Audio validation and XLS-R 300M embedding extraction reuse the existing project code. XLS-R stays frozen and runs locally using MPS when available. A `StandardScaler` plus `LogisticRegression` classifier is trained for the three personal-demo labels. Logistic regression is selected because the first usable dataset may have only one training recording per class after holding one recording per class out for testing; the main four-class classifier retains its separate calibrated SVC workflow.

The three labels are exactly:

- `positive`: a positive or happy vocal expression.
- `negative`: a negative, sad, angry, or frustrated vocal expression.
- `calm`: a calm or neutral vocal expression.

## Split and Metrics

At least two recordings per class are required. The application makes a deterministic stratified recording-level split. With exactly two recordings per class, one per class is used for training and one per class for testing. With more recordings, it holds out roughly one third, while ensuring at least one training recording per class.

The page reports:

- `same_speaker_accuracy`: correctly classified held-out recordings divided by held-out recordings.
- `same_speaker_macro_f1`: macro F1 across the three labels.
- number of recordings in each class and in the training/test subsets.

Every personal-demo metric is labelled as **same-speaker testing only**. It must not be presented as performance on new speakers.

## Result Contract

Successful analysis returns:

```json
{
  "status": "ok",
  "emotion": "positive",
  "scores": {
    "positive": 0.72,
    "negative": 0.18,
    "calm": 0.10
  },
  "latency_ms": 850.4,
  "model_version": "personal-three-emotion-v1"
}
```

The UI displays `scores` as percentages. Invalid audio or an unavailable model returns a safe Chinese error message and no emotional label.

## Error Handling

- Reject silent, unreadable, or shorter-than-0.5-second input using the existing audio validator.
- Do not allow model training until each label has two valid recordings.
- If XLS-R feature extraction fails, show a Chinese retry message.
- If no trained personal model exists, prompt the user to add recordings and train first.

## Testing

Unit tests use fake feature vectors, not private recordings or downloaded model calls. They verify:

- dataset counting and minimum-recording validation;
- stratified same-speaker split with two recordings per class;
- persisted model prediction returns all three score keys and a percentage-compatible probability distribution;
- error results omit an emotion label.

Manual verification records or uploads one local clip through the Gradio page, trains after adding six clips, and runs one analysis.

## Non-Goals

- No claim of general speaker-independent accuracy.
- No facial model fusion or Chatbot integration in this demo.
- No cloud deployment, authentication, or remote audio storage.
