from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from speech_emotion.integrated_demo import (
    AnalysisServices,
    analyse_recording,
    correct_combined_analysis,
)
from speech_emotion.speaker_identity import IdentityResult
from speech_emotion.transcription import TranscriptResult
from speech_emotion.workspace import ProjectWorkspace


@dataclass
class FakeIdentity:
    def identify(self, path: Path) -> IdentityResult:
        return IdentityResult("莊媗丞", {"莊媗丞": 0.9}, False, 1.0)


@dataclass
class FakeEmotion:
    def predict(self, path: Path) -> dict[str, object]:
        return {"status": "ok", "emotion": "calm", "scores": {"calm": 1.0}}


@dataclass
class FakeTranscriber:
    def transcribe(self, path: Path) -> str:
        return "這是一段測試"


def fake_services() -> AnalysisServices:
    return AnalysisServices(FakeIdentity(), FakeEmotion(), FakeTranscriber())


def test_analyse_recording_archives_once_and_returns_three_results(tmp_path, wav_factory):
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    result = analyse_recording(
        wav_factory("input.wav", np.ones(16_000, dtype=np.float32)),
        "upload",
        workspace,
        fake_services(),
    )

    assert result.status == "ok"
    assert result.archive_path.exists()
    assert result.identity.speaker == "莊媗丞"
    assert result.emotion["emotion"] == "calm"
    assert result.transcript is not None
    assert result.transcript.original_text == "這是一段測試"
    saved = json.loads((workspace.root / "05_完整分析紀錄/records" / f"{result.record_id}.json").read_text(encoding="utf-8"))
    assert saved["identity"]["speaker"] == result.identity.speaker
    assert saved["identity"]["scores"] == result.identity.scores


def test_invalid_audio_is_archived_but_not_sent_to_model(tmp_path):
    source = tmp_path / "broken.webm"
    source.write_bytes(b"not audio")
    workspace = ProjectWorkspace(tmp_path / "專題研究")

    result = analyse_recording(source, "upload", workspace, fake_services())

    assert result.status == "error"
    assert result.archive_path.exists()


def test_correction_updates_displayed_speaker_and_transcript(tmp_path, wav_factory):
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    original = analyse_recording(
        wav_factory("input.wav", np.ones(16_000, dtype=np.float32)),
        "upload",
        workspace,
        fake_services(),
    )

    corrected = correct_combined_analysis(
        original.record_id,
        speaker="紀律廷",
        emotion=None,
        transcript="正確文字",
        workspace=workspace,
        services=fake_services(),
    )

    assert corrected.identity is not None
    assert corrected.identity.speaker == "紀律廷"
    assert corrected.transcript is not None
    assert corrected.transcript.corrected_text == "正確文字"
