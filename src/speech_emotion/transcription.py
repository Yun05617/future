"""Local Chinese transcription storage, correction, and CER helpers."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

from speech_emotion.workspace import ProjectWorkspace
from speech_emotion.traditional_chinese import TraditionalChineseError, to_traditional_chinese


class TranscriptError(ValueError):
    """Raised when a transcript cannot be generated or safely updated."""


class ChineseTranscriber(Protocol):
    """Minimal adapter protocol for a local Chinese speech-to-text model."""

    def transcribe(self, audio_path: Path) -> str:
        """Return Chinese text for one local recording."""


@dataclass(frozen=True)
class TranscriptResult:
    """Original and optionally corrected transcript for one archived recording."""

    record_id: str
    original_text: str
    corrected_text: str | None
    cer: float | None
    latency_ms: float
    traditional_text: str | None = None

    @property
    def effective_text(self) -> str:
        """Normalize corrections and legacy raw-only records for display."""
        text = self.corrected_text or self.traditional_text or self.original_text
        return to_traditional_chinese(text) if text.strip() else text


def _normalize_text(text: str) -> str:
    return "".join(character for character in text.strip() if not character.isspace())


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return character error rate, or None until a non-empty correction exists."""
    reference_chars = list(_normalize_text(reference))
    hypothesis_chars = list(_normalize_text(hypothesis))
    if not reference_chars or not hypothesis_chars:
        return None
    previous = list(range(len(hypothesis_chars) + 1))
    for reference_index, reference_character in enumerate(reference_chars, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis_chars, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1] / len(reference_chars)


def _record_path(directory: Path, record_id: str) -> Path:
    if not record_id or Path(record_id).name != record_id:
        raise TranscriptError("轉寫紀錄編號不正確")
    return directory / f"{record_id}.json"


def _read_result(path: Path) -> TranscriptResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TranscriptResult(
            record_id=str(payload["record_id"]),
            original_text=str(payload["original_text"]),
            corrected_text=(
                str(payload["corrected_text"])
                if payload.get("corrected_text") is not None
                else None
            ),
            cer=(float(payload["cer"]) if payload.get("cer") is not None else None),
            latency_ms=float(payload["latency_ms"]),
            traditional_text=(
                str(payload["traditional_text"])
                if payload.get("traditional_text") is not None
                else None
            ),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise TranscriptError("轉寫紀錄無法讀取") from error


def transcribe_chinese(
    audio_path: Path,
    record_id: str,
    workspace: ProjectWorkspace,
    transcriber: ChineseTranscriber,
) -> TranscriptResult:
    """Create one local original Chinese transcript record."""
    started = perf_counter()
    try:
        original_text = transcriber.transcribe(audio_path).strip()
    except TranscriptError:
        raise
    except Exception as error:
        raise TranscriptError("中文語音轉文字失敗") from error
    if not original_text:
        raise TranscriptError("語音轉文字沒有產生內容")
    try:
        traditional_text = to_traditional_chinese(original_text)
    except TraditionalChineseError as error:
        raise TranscriptError("逐字稿無法轉為繁體中文，請確認本機中文支援") from error
    result = TranscriptResult(
        record_id=record_id,
        original_text=original_text,
        corrected_text=None,
        cer=None,
        latency_ms=float((perf_counter() - started) * 1000),
        traditional_text=traditional_text,
    )
    path = _record_path(
        workspace.root / "04_中文語音轉文字" / "原始轉寫結果", record_id
    )
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def save_transcript_correction(
    record_id: str, corrected_text: str, workspace: ProjectWorkspace
) -> TranscriptResult:
    """Append a corrected Chinese transcript while keeping original output intact."""
    original_path = _record_path(
        workspace.root / "04_中文語音轉文字" / "原始轉寫結果", record_id
    )
    original = _read_result(original_path)
    cleaned = corrected_text.strip()
    if cleaned:
        cleaned = to_traditional_chinese(cleaned)
    result = TranscriptResult(
        record_id=record_id,
        original_text=original.original_text,
        corrected_text=cleaned or None,
        cer=character_error_rate(cleaned, original.effective_text),
        latency_ms=original.latency_ms,
        traditional_text=original.traditional_text,
    )
    correction_path = _record_path(
        workspace.root / "04_中文語音轉文字" / "人工校正結果", record_id
    )
    correction_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


class OpenAIWhisperTranscriber:
    """Lazy local OpenAI Whisper adapter fixed to Chinese transcription."""

    def __init__(self, model_name: str = "small") -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: Path) -> str:
        try:
            import whisper
        except ImportError as error:
            raise TranscriptError("尚未安裝 Whisper 中文轉寫模型") from error
        if self._model is None:
            try:
                self._model = whisper.load_model(self.model_name)
            except Exception as error:
                raise TranscriptError(
                    "Whisper 模型尚未下載完成或無法載入，請確認網路與模型快取後再試"
                ) from error
        try:
            result = self._model.transcribe(str(audio_path), language="zh", fp16=False)
        except Exception as error:
            raise TranscriptError("Whisper 中文轉寫失敗，請確認音檔可正常播放") from error
        return str(result.get("text", ""))
