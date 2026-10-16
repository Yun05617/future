from pathlib import Path

import numpy as np
import pytest

from speech_emotion.transcription import (
    OpenAIWhisperTranscriber,
    TranscriptError,
    TranscriptResult,
    character_error_rate,
    save_transcript_correction,
    transcribe_chinese,
)
from speech_emotion.workspace import ProjectWorkspace


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio_path: Path) -> str:
        return self.text


class BrokenTranscriber:
    def transcribe(self, audio_path: Path) -> str:
        raise TranscriptError("Whisper 模型尚未下載完成或無法載入")


def test_transcription_persists_original_and_corrected_chinese_text(tmp_path, wav_factory):
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    result = transcribe_chinese(
        wav_factory("line.wav", np.ones(16_000, dtype=np.float32)),
        "record-1",
        workspace,
        FakeTranscriber("今天天氣很好"),
    )

    corrected = save_transcript_correction("record-1", "今天天氣非常好", workspace)

    assert result.original_text == "今天天氣很好"
    assert corrected.original_text == "今天天氣很好"
    assert corrected.corrected_text == "今天天氣非常好"
    assert corrected.cer is not None


def test_character_error_rate_handles_empty_corrected_text():
    assert character_error_rate("你好", "") is None


def test_effective_transcript_prefers_correction_then_traditional_text():
    result = TranscriptResult("r1", "我失恋了", None, None, 1.0, "我失戀了")
    assert result.effective_text == "我失戀了"

    corrected = TranscriptResult("r1", "我失恋了", "我已經走出來了", None, 1.0, "我失戀了")
    assert corrected.effective_text == "我已經走出來了"


def test_legacy_and_corrected_transcripts_are_always_traditional():
    assert TranscriptResult("r1", "我失恋了", None, None, 1).effective_text == "我失戀了"
    assert TranscriptResult("r1", "原文", "听得到吗", None, 1).effective_text == "聽得到嗎"


def test_simplified_transcription_keeps_raw_but_saves_traditional(tmp_path):
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    result = transcribe_chinese(Path("voice.wav"), "r1", workspace, FakeTranscriber("听得到吗"))
    corrected = save_transcript_correction("r1", "我听到了", workspace)
    assert result.original_text == "听得到吗"
    assert result.traditional_text == "聽得到嗎"
    assert corrected.corrected_text == "我聽到了"
    assert corrected.original_text == "听得到吗"
    unchanged = save_transcript_correction("r1", "聽得到嗎", workspace)
    assert unchanged.cer == 0.0


def test_conversion_failure_does_not_silently_display_simplified(monkeypatch, tmp_path):
    from speech_emotion import transcription
    from speech_emotion.traditional_chinese import TraditionalChineseError

    def unavailable(_):
        raise TraditionalChineseError("unavailable")

    monkeypatch.setattr(transcription, "to_traditional_chinese", unavailable)
    with pytest.raises(TranscriptError, match="繁體中文"):
        transcribe_chinese(Path("voice.wav"), "r1", ProjectWorkspace(tmp_path), FakeTranscriber("语音"))


def test_openai_whisper_defaults_to_small_model():
    assert OpenAIWhisperTranscriber().model_name == "small"


def test_transcribe_chinese_reports_model_load_failure(tmp_path):
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    with pytest.raises(TranscriptError, match="Whisper 模型"):
        transcribe_chinese(tmp_path / "voice.wav", "r1", workspace, BrokenTranscriber())
