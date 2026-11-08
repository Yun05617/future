import shutil
import subprocess

import numpy as np
import pytest
import soundfile

from speech_emotion.audio import (
    AudioValidationError,
    denoise_and_normalize_audio,
    load_valid_audio,
    normalize_audio_to_wav,
)


def test_rejects_silent_audio(wav_factory):
    path = wav_factory("silent.wav", np.zeros(16000, dtype=np.float32))
    with pytest.raises(AudioValidationError, match="silent"):
        load_valid_audio(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_normalizes_mp3_to_valid_16khz_mono_wav(tmp_path, wav_factory):
    source_wav = wav_factory("source.wav", np.ones(32_000, dtype=np.float32), 32_000)
    source_mp3 = tmp_path / "source.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source_wav), str(source_mp3)],
        check=True,
        capture_output=True,
    )
    destination = tmp_path / "normalized.wav"

    normalize_audio_to_wav(source_mp3, destination)

    samples, sample_rate = load_valid_audio(destination)
    assert destination.exists()
    assert destination.read_bytes()[:4] == b"RIFF"
    assert sample_rate == 16_000
    assert len(samples) >= 16_000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_denoise_and_normalize_creates_valid_16khz_mono_wav(tmp_path, wav_factory):
    source = wav_factory("input.wav", np.ones(32_000, dtype=np.float32), 32_000)

    output = denoise_and_normalize_audio(source, tmp_path / "derived.wav")

    samples, sample_rate = soundfile.read(output)
    assert output.exists()
    assert sample_rate == 16_000
    assert samples.ndim == 1


def test_denoise_and_normalize_removes_partial_output_on_invalid_audio(tmp_path):
    destination = tmp_path / "derived.wav"

    with pytest.raises(AudioValidationError):
        denoise_and_normalize_audio(tmp_path / "missing.webm", destination)

    assert not destination.exists()
