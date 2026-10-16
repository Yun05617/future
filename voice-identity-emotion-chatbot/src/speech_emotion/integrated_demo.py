"""Combine local archive, denoise, identity, emotion, and transcript results."""

from dataclasses import dataclass
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Protocol

from speech_emotion.audio import AudioValidationError, denoise_and_normalize_audio
from speech_emotion.speaker_identity import IdentityResult, KNOWN_SPEAKERS, UNKNOWN_SPEAKER
from speech_emotion.transcription import TranscriptResult, character_error_rate, transcribe_chinese
from speech_emotion.workspace import ArchivedAudio, ProjectWorkspace, WorkspaceError


logger = logging.getLogger(__name__)


class IdentityService(Protocol):
    def identify(self, path: Path) -> IdentityResult: ...


class EmotionService(Protocol):
    def predict(self, path: Path) -> dict[str, object]: ...


class TranscriptService(Protocol):
    def transcribe(self, path: Path) -> str: ...


@dataclass(frozen=True)
class AnalysisServices:
    identity: IdentityService
    emotion: EmotionService
    transcriber: TranscriptService


@dataclass(frozen=True)
class CombinedAnalysis:
    record_id: str
    archive_path: Path
    processed_path: Path | None
    identity: IdentityResult | None
    emotion: dict[str, object] | None
    transcript: TranscriptResult | None
    total_latency_ms: float
    status: str
    message: str | None = None


def _processed_path(workspace: ProjectWorkspace, record_id: str) -> Path:
    directory = workspace.root / "05_完整分析紀錄" / "處理後音檔"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{record_id}.wav"


def _record_path(workspace: ProjectWorkspace, record_id: str) -> Path:
    directory = workspace.root / "05_完整分析紀錄" / "records"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{record_id}.json"


def _payload(result: CombinedAnalysis) -> dict[str, object]:
    return {
        "record_id": result.record_id,
        "archive_path": str(result.archive_path),
        "processed_path": str(result.processed_path) if result.processed_path else None,
        "identity": (
            {"speaker": result.identity.speaker, "scores": result.identity.scores, "is_unknown": result.identity.is_unknown, "latency_ms": result.identity.latency_ms}
            if result.identity else None
        ),
        "emotion": result.emotion,
        "transcript": (
            {"record_id": result.transcript.record_id, "original_text": result.transcript.original_text, "traditional_text": result.transcript.traditional_text, "corrected_text": result.transcript.corrected_text, "cer": result.transcript.cer, "latency_ms": result.transcript.latency_ms}
            if result.transcript else None
        ),
        "total_latency_ms": result.total_latency_ms,
        "status": result.status,
        "message": result.message,
    }


def _save_analysis(workspace: ProjectWorkspace, result: CombinedAnalysis) -> None:
    _record_path(workspace, result.record_id).write_text(
        json.dumps(_payload(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    workspace.write_json(
        "05_完整分析紀錄",
        {
            "record_id": result.record_id,
            "archive_path": str(result.archive_path),
            "processed_path": str(result.processed_path) if result.processed_path else None,
            "identity": result.identity.speaker if result.identity else None,
            "emotion": result.emotion.get("emotion") if result.emotion else None,
            "transcript": result.transcript.effective_text if result.transcript else None,
            "status": result.status,
            "message": result.message,
        },
    )


def analyse_recording(
    source: Path,
    source_kind: str,
    workspace: ProjectWorkspace,
    services: AnalysisServices,
) -> CombinedAnalysis:
    """Archive first, then run all local analysers on one denoised derivative."""
    started = perf_counter()
    archived: ArchivedAudio = workspace.archive_input(source, source_kind)
    try:
        processed = denoise_and_normalize_audio(
            archived.original_path, _processed_path(workspace, archived.record_id)
        )
        # These three analysers read the same denoised file and do not depend on
        # one another. Running them together removes the sum of their cold-start
        # latencies from a voice turn (the local models remain fully offline).
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="voice-analysis") as pool:
            identity_future = pool.submit(services.identity.identify, processed)
            emotion_future = pool.submit(services.emotion.predict, processed)
            transcript_future = pool.submit(
                transcribe_chinese, processed, archived.record_id, workspace, services.transcriber
            )
            identity = identity_future.result()
            emotion = emotion_future.result()
            transcript = transcript_future.result()
        result = CombinedAnalysis(
            archived.record_id,
            archived.original_path,
            processed,
            identity,
            emotion,
            transcript,
            float((perf_counter() - started) * 1000),
            "ok",
        )
    except (AudioValidationError, WorkspaceError, ValueError) as error:
        result = CombinedAnalysis(
            archived.record_id,
            archived.original_path,
            None,
            None,
            None,
            None,
            float((perf_counter() - started) * 1000),
            "error",
            str(error),
        )
    except Exception as error:
        logger.exception("Voice analysis failed for recording %s", archived.record_id)
        result = CombinedAnalysis(
            archived.record_id,
            archived.original_path,
            None,
            None,
            None,
            None,
            float((perf_counter() - started) * 1000),
            "error",
            f"語音模型無法完成分析（{type(error).__name__}），請確認本機模型已載入後重試。",
        )
    _save_analysis(workspace, result)
    return result


def correct_combined_analysis(
    record_id: str,
    speaker: str | None,
    emotion: str | None,
    transcript: str | None,
    workspace: ProjectWorkspace,
    services: AnalysisServices,
) -> CombinedAnalysis:
    """Persist display corrections for one prior analysis without deleting its raw result."""
    del services
    try:
        payload = json.loads(_record_path(workspace, record_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("找不到可更正的分析紀錄") from error
    identity_data = payload.get("identity")
    transcript_data = payload.get("transcript")
    if not isinstance(identity_data, dict) or not isinstance(transcript_data, dict):
        raise ValueError("這筆分析紀錄無法更正")
    final_speaker = speaker or str(identity_data["speaker"])
    if final_speaker not in {*KNOWN_SPEAKERS, UNKNOWN_SPEAKER}:
        raise ValueError("不支援的成員名稱")
    identity = IdentityResult(
        final_speaker,
        {str(key): float(value) for key, value in dict(identity_data["scores"]).items()},
        final_speaker == UNKNOWN_SPEAKER,
        float(identity_data["latency_ms"]),
    )
    original_text = str(transcript_data["original_text"])
    corrected_text = transcript.strip() if transcript is not None and transcript.strip() else None
    corrected_transcript = TranscriptResult(
        record_id,
        original_text,
        corrected_text,
        character_error_rate(corrected_text or "", original_text),
        float(transcript_data["latency_ms"]),
        transcript_data.get("traditional_text"),
    )
    updated_emotion = dict(payload.get("emotion") or {})
    if emotion is not None:
        updated_emotion["emotion"] = emotion
    result = CombinedAnalysis(
        record_id,
        Path(str(payload["archive_path"])),
        Path(str(payload["processed_path"])) if payload.get("processed_path") else None,
        identity,
        updated_emotion,
        corrected_transcript,
        float(payload["total_latency_ms"]),
        str(payload["status"]),
        payload.get("message"),
    )
    _save_analysis(workspace, result)
    return result
