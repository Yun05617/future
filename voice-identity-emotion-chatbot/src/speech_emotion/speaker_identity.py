"""Local speaker enrollment and identity matching for the four project members."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol
import shutil
from uuid import uuid4

import numpy as np
import torch

from speech_emotion.audio import load_valid_audio
from speech_emotion.workspace import ProjectWorkspace, WorkspaceError


KNOWN_SPEAKERS = ("莊媗丞", "姜昀廷", "紀律廷", "趙科翰")
UNKNOWN_SPEAKER = "未知使用者"
IDENTITY_THRESHOLD = 0.40
IDENTITY_MARGIN = 0.0


class SpeakerIdentityError(ValueError):
    """Raised when local speaker enrollment or matching cannot run safely."""


class SpeakerEmbedder(Protocol):
    """Minimal interface allowing production and fake voice embedding models."""

    def embed(self, path: Path) -> np.ndarray:
        """Return one non-empty voice embedding for an audio file."""


@dataclass(frozen=True)
class IdentityResult:
    """Local identity scores and the final known-or-unknown decision."""

    speaker: str
    scores: dict[str, float]
    is_unknown: bool
    latency_ms: float


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return a finite unit vector or reject invalid model output."""
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not len(values) or not np.isfinite(values).all() or norm == 0.0:
        raise SpeakerIdentityError("聲紋特徵無法使用")
    return values / norm


def _enrollment_directory(workspace: ProjectWorkspace, speaker: str) -> Path:
    if speaker not in KNOWN_SPEAKERS:
        raise SpeakerIdentityError("不支援的成員名稱")
    return workspace.root / "01_原始音檔" / "身分辨識" / speaker


def _centroid_path(workspace: ProjectWorkspace, speaker: str) -> Path:
    return workspace.root / "03_語音身分辨識" / "成員聲紋範本" / f"{speaker}.npy"


def enroll_speaker(audio_path: Path, speaker: str, workspace: ProjectWorkspace) -> Path:
    """Copy one processed recording into the selected member's local enrollment set."""
    if not audio_path.is_file():
        raise SpeakerIdentityError("找不到要加入的錄音檔")
    directory = _enrollment_directory(workspace, speaker)
    destination = directory / f"{uuid4().hex}.wav"
    shutil.copy2(audio_path, destination)
    return destination


def rebuild_speaker_profiles(
    workspace: ProjectWorkspace, embedder: SpeakerEmbedder
) -> dict[str, int]:
    """Build normalized speaker centroids from every enrolled local WAV file."""
    counts: dict[str, int] = {}
    vector_directory = workspace.root / "03_語音身分辨識" / "聲紋向量"
    vector_directory.mkdir(parents=True, exist_ok=True)
    for speaker in KNOWN_SPEAKERS:
        recordings = sorted(_enrollment_directory(workspace, speaker).glob("*.wav"))
        counts[speaker] = len(recordings)
        centroid_path = _centroid_path(workspace, speaker)
        if not recordings:
            centroid_path.unlink(missing_ok=True)
            continue
        vectors = [_unit(embedder.embed(recording)) for recording in recordings]
        for recording, vector in zip(recordings, vectors):
            np.save(vector_directory / f"{recording.stem}.npy", vector)
        centroid = _unit(np.mean(vectors, axis=0))
        np.save(centroid_path, centroid)
    return counts


def _load_centroid(workspace: ProjectWorkspace, speaker: str) -> np.ndarray | None:
    path = _centroid_path(workspace, speaker)
    if not path.is_file():
        return None
    try:
        return _unit(np.load(path, allow_pickle=False))
    except (OSError, ValueError) as error:
        raise SpeakerIdentityError("成員聲紋範本無法讀取") from error


def identify_speaker(
    audio_path: Path,
    workspace: ProjectWorkspace,
    embedder: SpeakerEmbedder,
    threshold: float,
    margin: float,
) -> IdentityResult:
    """Accept the highest centroid score strictly above the supplied threshold."""
    started = perf_counter()
    if not 0.0 <= threshold <= 1.0 or margin < 0.0:
        raise SpeakerIdentityError("身分辨識門檻設定不正確")
    vector = _unit(embedder.embed(audio_path))
    scores: dict[str, float] = {}
    for speaker in KNOWN_SPEAKERS:
        centroid = _load_centroid(workspace, speaker)
        scores[speaker] = float(np.dot(vector, centroid)) if centroid is not None else -1.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_score = ranked[0]
    second_score = ranked[1][1]
    # Scores are float32; compare at the same precision to keep 40% exclusive.
    is_unknown = best_score <= float(np.float32(threshold)) or best_score - second_score < margin
    return IdentityResult(
        speaker=UNKNOWN_SPEAKER if is_unknown else best_speaker,
        scores=scores,
        is_unknown=is_unknown,
        latency_ms=float((perf_counter() - started) * 1000),
    )


class SpeechBrainEmbedder:
    """Lazy SpeechBrain adapter so imports and tests do not download model weights."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        self.source = source
        self._classifier = None

    def embed(self, path: Path) -> np.ndarray:
        """Load the approved local/pretrained model on first real inference."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as error:
            raise SpeakerIdentityError("尚未安裝 SpeechBrain 聲紋模型") from error
        if self._classifier is None:
            self._classifier = EncoderClassifier.from_hparams(source=self.source)
        samples, _ = load_valid_audio(path)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        embedding = self._classifier.encode_batch(waveform)
        return np.asarray(embedding.squeeze().detach().cpu().numpy(), dtype=np.float32)
