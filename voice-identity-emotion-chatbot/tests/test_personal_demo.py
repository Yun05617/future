import json

import numpy as np
import joblib
import pandas as pd
import pytest
import sys
import speech_emotion.personal_demo_ui as personal_demo_ui
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from speech_emotion.personal_demo import (
    PersonalDemoError,
    add_training_audio,
    add_manual_training_audio,
    correct_analysis_and_retrain,
    predict_personal_demo,
    train_personal_demo,
    training_counts,
    undo_last_manual_training_audio,
    undo_last_correction_and_retrain,
)
from speech_emotion.personal_demo_ui import (
    format_analysis,
    select_input_audio,
    select_training_audio_paths,
)
from speech_emotion.integrated_demo import CombinedAnalysis
from speech_emotion.speaker_identity import IdentityResult
from speech_emotion.transcription import TranscriptResult


_RUNNER_SPEC = spec_from_file_location(
    "run_personal_demo", Path(__file__).parents[1] / "scripts" / "run_personal_demo.py"
)
assert _RUNNER_SPEC and _RUNNER_SPEC.loader
run_personal_demo = module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(run_personal_demo)


@pytest.fixture
def personal_model(tmp_path):
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )
    training_features = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], columns=["f1", "f2"]
    )
    pipeline.fit(
        training_features,
        ["positive", "negative", "calm"],
    )
    model_path = tmp_path / "personal.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": ["f1", "f2"],
            "labels": ["positive", "negative", "calm"],
            "model_version": "personal-three-emotion-v1",
        },
        model_path,
    )
    return model_path


def test_predict_personal_demo_returns_all_three_probabilities(
    monkeypatch, tmp_path, personal_model
):
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda _: pd.Series({"f2": 0.3, "f1": 0.2}),
    )

    result = predict_personal_demo(tmp_path / "clip.wav", personal_model)

    assert result["status"] == "ok"
    assert set(result["scores"]) == {"positive", "negative", "calm"}
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6
    assert result["emotion"] in result["scores"]
    assert result["model_version"] == "personal-three-emotion-v1"
    assert isinstance(result["latency_ms"], float)


def test_combined_analysis_format_uses_effective_traditional_transcript():
    analysis = CombinedAnalysis(
        "record-1", Path("raw.wav"), Path("processed.wav"),
        IdentityResult("莊媗丞", {"莊媗丞": 0.9}, False, 1.0),
        {"emotion": "calm", "scores": {"calm": 1.0}},
        TranscriptResult("record-1", "我失恋了", None, None, 1.0, "我失戀了"),
        1.0, "ok",
    )
    formatted = personal_demo_ui._format_combined_analysis(analysis)
    assert "我失戀了" in formatted[3]
    assert "我失恋了" not in formatted[3]


def test_predict_personal_demo_returns_safe_error_without_model(tmp_path):
    result = predict_personal_demo(
        tmp_path / "clip.wav", tmp_path / "missing.joblib"
    )

    assert result["status"] == "error"
    assert "emotion" not in result
    assert "scores" not in result
    assert isinstance(result["message"], str)
    assert isinstance(result["latency_ms"], float)


def test_predict_personal_demo_returns_safe_error_for_silent_audio(
    tmp_path, wav_factory, personal_model
):
    silent = wav_factory("silent.wav", np.zeros(16_000, dtype=np.float32))

    result = predict_personal_demo(silent, personal_model)

    assert result["status"] == "error"
    assert "emotion" not in result
    assert "scores" not in result


def test_add_training_audio_copies_valid_clip_and_counts_each_label(tmp_path, wav_factory):
    source = wav_factory("positive.wav", np.ones(16000, dtype=np.float32))

    counts = add_training_audio(source, "positive", tmp_path / "recordings")

    assert counts == {"positive": 1, "negative": 0, "calm": 0}
    assert list((tmp_path / "recordings" / "positive").glob("*.wav"))


def test_add_training_audio_rejects_silent_clip_without_storing_or_counting(
    tmp_path, wav_factory
):
    recordings_root = tmp_path / "recordings"
    silent = wav_factory("silent.wav", np.zeros(16_000, dtype=np.float32))

    with pytest.raises(PersonalDemoError, match="無法使用"):
        add_training_audio(silent, "positive", recordings_root)

    assert training_counts(recordings_root) == {
        "positive": 0,
        "negative": 0,
        "calm": 0,
    }
    assert not recordings_root.exists()


def test_add_training_audio_rejects_too_short_clip_without_storing_or_counting(
    tmp_path, wav_factory
):
    recordings_root = tmp_path / "recordings"
    short = wav_factory("short.wav", np.ones(7_999, dtype=np.float32))

    with pytest.raises(PersonalDemoError, match="無法使用"):
        add_training_audio(short, "positive", recordings_root)

    assert training_counts(recordings_root) == {
        "positive": 0,
        "negative": 0,
        "calm": 0,
    }
    assert not recordings_root.exists()


def test_undo_last_manual_training_audio_removes_only_the_latest_manual_clip(
    tmp_path, wav_factory
):
    recordings_root = tmp_path / "data" / "raw" / "personal_demo"
    first_clip = wav_factory("first.wav", np.ones(16_000, dtype=np.float32))
    second_clip = wav_factory("second.wav", np.full(16_000, 0.2, dtype=np.float32))

    add_manual_training_audio(first_clip, "positive", recordings_root)
    add_manual_training_audio(second_clip, "negative", recordings_root)
    counts = undo_last_manual_training_audio(recordings_root)

    assert counts == {"positive": 1, "negative": 0, "calm": 0}
    assert len(list((recordings_root / "positive").glob("*.wav"))) == 1
    assert not list((recordings_root / "negative").glob("*.wav"))
    history_path = (
        tmp_path / "data" / "processed" / "personal_demo" / "manual_addition_history.json"
    )
    assert len(json.loads(history_path.read_text(encoding="utf-8"))) == 1


def test_trains_with_two_clips_per_label_and_reports_same_speaker_metrics(
    monkeypatch, tmp_path
):
    root = tmp_path / "recordings"
    for label in ("positive", "negative", "calm"):
        for index in range(2):
            path = root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda path: pd.Series(
            {
                "f1": float(path.parent.name == "positive"),
                "f2": float(path.parent.name == "negative"),
            }
        ),
    )

    result = train_personal_demo(root, tmp_path / "personal.joblib")

    assert result["same_speaker_test_count"] == 3
    assert isinstance(result["same_speaker_accuracy"], float)
    assert (tmp_path / "personal.joblib").exists()


def test_training_writes_readable_metadata_with_model_labels_and_split(
    monkeypatch, tmp_path
):
    recordings_root = tmp_path / "data" / "raw" / "personal_demo"
    for label in ("positive", "negative", "calm"):
        for index in range(2):
            path = recordings_root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda path: pd.Series(
            {
                "f1": float(path.parent.name == "positive"),
                "f2": float(path.parent.name == "negative"),
            }
        ),
    )

    train_personal_demo(recordings_root, tmp_path / "personal.joblib")

    metadata_path = (
        tmp_path
        / "data"
        / "processed"
        / "personal_demo"
        / "training_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_version"] == "personal-three-emotion-v1"
    assert metadata["labels"] == ["calm", "negative", "positive"]
    assert metadata["split_paths"] == {
        "train": [
            str(recordings_root / "positive" / "1.wav"),
            str(recordings_root / "negative" / "1.wav"),
            str(recordings_root / "calm" / "1.wav"),
        ],
        "test": [
            str(recordings_root / "positive" / "0.wav"),
            str(recordings_root / "negative" / "0.wav"),
            str(recordings_root / "calm" / "0.wav"),
        ],
    }


def test_rejects_training_when_a_label_has_fewer_than_two_clips(tmp_path):
    root = tmp_path / "recordings"
    for label, count in {"positive": 2, "negative": 2, "calm": 1}.items():
        for index in range(count):
            path = root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    with pytest.raises(PersonalDemoError, match="至少 2 段"):
        train_personal_demo(root, tmp_path / "personal.joblib")


def test_corrected_analysis_is_stored_with_its_label_and_retrains(
    monkeypatch, tmp_path, wav_factory
):
    recordings_root = tmp_path / "data" / "raw" / "personal_demo"
    for label in ("positive", "negative", "calm"):
        for index in range(2):
            path = recordings_root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            wav_factory(
                str(path.relative_to(tmp_path)),
                np.full(16_000, 0.1 + index * 0.01, dtype=np.float32),
            )
    corrected_clip = wav_factory(
        "analysis.wav", np.full(16_000, 0.3, dtype=np.float32)
    )
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda path: pd.Series(
            {
                "f1": float(path.parent.name == "positive"),
                "f2": float(path.parent.name == "negative"),
            }
        ),
    )

    result = correct_analysis_and_retrain(
        corrected_clip,
        "negative",
        recordings_root,
        tmp_path / "models" / "personal_demo.joblib",
    )

    assert result["counts"] == {"positive": 2, "negative": 3, "calm": 2}
    assert result["same_speaker_train_count"] == 4
    assert result["same_speaker_test_count"] == 3
    assert len(list((recordings_root / "negative").glob("*.wav"))) == 3
    assert (tmp_path / "models" / "personal_demo.joblib").exists()


def test_undo_last_correction_removes_only_that_clip_and_retrains(
    monkeypatch, tmp_path, wav_factory
):
    recordings_root = tmp_path / "data" / "raw" / "personal_demo"
    for label in ("positive", "negative", "calm"):
        for index in range(2):
            path = recordings_root / label / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            wav_factory(
                str(path.relative_to(tmp_path)),
                np.full(16_000, 0.1 + index * 0.01, dtype=np.float32),
            )
    corrected_clip = wav_factory(
        "analysis.wav", np.full(16_000, 0.3, dtype=np.float32)
    )
    model_path = tmp_path / "models" / "personal_demo.joblib"
    monkeypatch.setattr(
        "speech_emotion.personal_demo.features_for_path",
        lambda path: pd.Series(
            {
                "f1": float(path.parent.name == "positive"),
                "f2": float(path.parent.name == "negative"),
            }
        ),
    )
    correct_analysis_and_retrain(
        corrected_clip, "negative", recordings_root, model_path
    )

    result = undo_last_correction_and_retrain(recordings_root, model_path)

    assert result["counts"] == {"positive": 2, "negative": 2, "calm": 2}
    assert result["same_speaker_train_count"] == 3
    assert result["same_speaker_test_count"] == 3
    assert len(list((recordings_root / "negative").glob("*.wav"))) == 2
    history_path = tmp_path / "data" / "processed" / "personal_demo" / "correction_history.json"
    assert json.loads(history_path.read_text(encoding="utf-8")) == []


def test_format_analysis_displays_label_percentages_and_latency():
    output = format_analysis(
        {
            "status": "ok",
            "emotion": "positive",
            "scores": {"positive": 0.72, "negative": 0.18, "calm": 0.10},
            "latency_ms": 850.4,
            "model_version": "personal-three-emotion-v1",
        }
    )

    assert "最終判斷：正面" in output
    assert "正面：72.0%" in output
    assert "分析時間：850.4 ms" in output


def test_select_input_audio_prefers_uploaded_file_and_keeps_microphone_fallback():
    assert select_input_audio("recording.wav", "sample.webm") == "sample.webm"
    assert select_input_audio("recording.wav", None) == "recording.wav"
    assert select_input_audio(None, None) is None


def test_select_training_audio_paths_keeps_all_uploaded_files_for_one_label():
    assert select_training_audio_paths("recording.wav", ["one.webm", "two.webm"]) == [
        "one.webm",
        "two.webm",
    ]
    assert select_training_audio_paths("recording.wav", None) == ["recording.wav"]
    assert select_training_audio_paths(None, None) == []


def test_source_kind_for_training_recording_uses_supported_archive_values():
    assert personal_demo_ui.source_kind_for_training_recording(
        "recording.wav", None
    ) == "microphone"
    assert personal_demo_ui.source_kind_for_training_recording(
        "recording.wav", ["one.wav"]
    ) == "upload"


def test_runner_keeps_server_local_and_does_not_accept_host_override(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDemo:
        def launch(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(run_personal_demo, "build_demo", lambda: FakeDemo())
    monkeypatch.setattr(sys, "argv", ["run_personal_demo.py", "--port", "8123"])

    run_personal_demo.main()

    assert captured["server_name"] == "127.0.0.1"
    assert captured["server_port"] == 8123
    assert captured["share"] is False
    assert captured["inbrowser"] is True
    assert captured["css"] == run_personal_demo.APPLE_STYLE_CSS
    assert captured["theme"] is run_personal_demo.APPLE_THEME

    monkeypatch.setattr(sys, "argv", ["run_personal_demo.py", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        run_personal_demo.parse_args()


def test_manual_documents_personal_demo_runner():
    manual = Path("docs/語音情緒辨識使用手冊.md").read_text(encoding="utf-8")

    assert "run_personal_demo.py" in manual
    assert "同一位說話者測試" in manual
