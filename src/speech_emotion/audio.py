"""Audio normalization, loading, and quality checks for one utterance."""

from pathlib import Path
import shutil
import subprocess
import tempfile

import librosa
import numpy as np


class AudioValidationError(ValueError):
    """Raised when an audio file cannot be used for emotion analysis."""


def _load_with_librosa(
    path: Path, target_sample_rate: int
) -> tuple[np.ndarray, int]:
    """Load and resample one locally readable audio file."""
    samples, sample_rate = librosa.load(
        path, sr=target_sample_rate, mono=True, dtype=np.float32
    )
    return samples.astype(np.float32, copy=False), sample_rate


def _validate_audio(
    samples: np.ndarray, sample_rate: int, min_duration_s: float
) -> None:
    """Reject recordings that are too short or effectively silent."""
    if len(samples) < min_duration_s * sample_rate:
        raise AudioValidationError("audio is shorter than the minimum duration")
    if float(np.max(np.abs(samples))) < 1e-4:
        raise AudioValidationError("audio is silent")


def _run_ffmpeg(
    source_path: Path, destination_path: Path, audio_filter: str | None = None
) -> None:
    """Convert common local audio formats to a mono, 16 kHz PCM WAV file."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        bundled_ffmpeg = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg.exe"
        if bundled_ffmpeg.is_file():
            ffmpeg_path = str(bundled_ffmpeg)
    if ffmpeg_path is None:
        raise AudioValidationError("ffmpeg is required to convert this audio format")
    try:
        command = [
            ffmpeg_path,
            "-nostdin",
            "-y",
            "-i",
            str(source_path),
        ]
        if audio_filter is not None:
            command.extend(["-af", audio_filter])
        command.extend(
            [
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination_path),
            ]
        )
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AudioValidationError("unable to convert audio") from error


def normalize_audio_to_wav(source_path: Path, destination_path: Path) -> None:
    """Create a validated 16 kHz mono WAV copy from a common audio format."""
    try:
        with tempfile.TemporaryDirectory(prefix="speech-emotion-") as directory:
            temporary_wav = Path(directory) / "normalized.wav"
            _run_ffmpeg(source_path, temporary_wav)
            samples, sample_rate = _load_with_librosa(temporary_wav, 16000)
            _validate_audio(samples, sample_rate, 0.5)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temporary_wav, destination_path)
    except Exception as error:
        destination_path.unlink(missing_ok=True)
        if isinstance(error, AudioValidationError):
            raise
        raise AudioValidationError("unable to normalize audio") from error


def denoise_and_normalize_audio(source_path: Path, destination_path: Path) -> Path:
    """Create a validated denoised derivative without modifying the original."""
    try:
        with tempfile.TemporaryDirectory(prefix="speech-emotion-") as directory:
            temporary_wav = Path(directory) / "denoised.wav"
            _run_ffmpeg(source_path, temporary_wav, audio_filter="afftdn=nf=-25")
            samples, sample_rate = _load_with_librosa(temporary_wav, 16000)
            _validate_audio(samples, sample_rate, 0.5)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temporary_wav, destination_path)
    except Exception as error:
        destination_path.unlink(missing_ok=True)
        if isinstance(error, AudioValidationError):
            raise
        raise AudioValidationError("unable to denoise and normalize audio") from error
    return destination_path


def load_valid_audio(
    path: Path,
    min_duration_s: float = 0.5,
    target_sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """Load common audio input and reject unusable recordings."""
    try:
        samples, sample_rate = _load_with_librosa(path, target_sample_rate)
    except Exception:
        with tempfile.TemporaryDirectory(prefix="speech-emotion-") as directory:
            temporary_wav = Path(directory) / "normalized.wav"
            _run_ffmpeg(path, temporary_wav)
            try:
                samples, sample_rate = _load_with_librosa(
                    temporary_wav, target_sample_rate
                )
            except Exception as error:
                raise AudioValidationError(f"unable to read audio: {path}") from error

    _validate_audio(samples, sample_rate, min_duration_s)
    return samples, sample_rate
