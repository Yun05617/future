"""Export traceable pitch-shift experiments without enrolling synthetic voices."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import librosa
import numpy as np
import soundfile as sf

from speech_emotion.audio import AudioValidationError, load_valid_audio


class PitchAugmentationError(ValueError):
    """Raised when an experimental pitch export cannot be created."""


@dataclass(frozen=True)
class PitchExport:
    archive_path: Path
    manifest_path: Path
    audio_paths: tuple[Path, ...]
    semitones: int


def _validate_steps(semitones: float) -> int:
    if isinstance(semitones, bool):
        raise PitchAugmentationError("半音數必須是 -12 至 12 之間的非零整數。")
    try:
        steps = float(semitones)
    except (TypeError, ValueError) as error:
        raise PitchAugmentationError("請提供有效的半音數。") from error
    if not math.isfinite(steps) or not steps.is_integer() or not 0 < abs(steps) <= 12:
        raise PitchAugmentationError("半音數必須是 -12 至 12 之間的非零整數。")
    return int(steps)


def export_pitch_batch(
    sources: list[Path], output_root: Path, semitones: float = 4,
    emotion_label: str | None = None,
) -> PitchExport:
    """Create one shifted copy per unique source; never update training data."""
    steps = _validate_steps(semitones)
    if not sources:
        raise PitchAugmentationError("請先錄音或上傳音訊檔。")
    if emotion_label not in {None, "positive", "negative", "calm"}:
        raise PitchAugmentationError("情緒標籤無法使用。")

    # Validate every source before creating output; file hashes also group variants.
    prepared = []
    seen = set()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise PitchAugmentationError("找不到部分來源音檔，請重新選取。")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen:
            continue
        try:
            samples, rate = load_valid_audio(source)
        except AudioValidationError as error:
            raise PitchAugmentationError("來源音檔無法使用，請確認不是靜音或過短。") from error
        if not np.isfinite(samples).all():
            raise PitchAugmentationError("來源音檔含有無效的取樣數值。")
        prepared.append((source, digest, samples, rate))
        seen.add(digest)

    output_root = Path(output_root).expanduser().resolve()
    if any(source.is_relative_to(output_root) for source, *_ in prepared):
        raise PitchAugmentationError("請選取原始錄音，不要在同一擴增資料夾內反覆變調。")
    directory = output_root / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    originals_directory = directory / 'originals'
    originals_directory.mkdir()
    entries = []
    audio_paths = []
    for source, digest, samples, rate in prepared:
        backup = originals_directory / f'{digest}{source.suffix or ".audio"}'
        shutil.copy2(source, backup)
        shifted = librosa.effects.pitch_shift(
            samples, sr=rate, n_steps=steps, bins_per_octave=12,
            res_type="soxr_hq", scale=False,
        )
        shifted = np.asarray(shifted, dtype=np.float32)
        if shifted.shape != samples.shape or not np.isfinite(shifted).all():
            raise PitchAugmentationError("變調結果無法使用；未將任何音檔加入訓練資料。")
        peak = float(np.max(np.abs(shifted)))
        gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
        shifted = shifted * gain
        filename = f"{digest[:16]}__pitch_{steps:+d}.wav"
        destination = directory / filename
        with destination.open('xb') as stream:
            sf.write(stream, shifted, rate, format='WAV', subtype='PCM_16')
        audio_paths.append(destination)
        entries.append({
            "source_name": source.name,
            "source_sha256": digest,
            "source_backup": backup.relative_to(directory).as_posix(),
            "group_id": digest,
            "output_name": filename,
            "sample_rate": rate,
            "samples": len(shifted),
            "semitones": steps,
            "output_gain": gain,
            "emotion_label_from_user": emotion_label,
            "identity_preserved": "not_verified",
            "emotion_preserved": "not_verified",
            "split": "unassigned",
        })
    manifest_path = directory / 'manifest.json'
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "experimental_pitch_augmentation",
        "method": "librosa.effects.pitch_shift",
        "semitones": steps,
        "originals_modified": False,
        "automatically_enrolled_or_trained": False,
        "split_policy": "Split originals first; augment training only; keep each group_id in one split.",
        "items": entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    archive_path = directory / 'pitch-augmentation.zip'
    with ZipFile(archive_path, 'x', compression=ZIP_DEFLATED) as archive:
        archive.write(manifest_path, manifest_path.name)
        for audio in audio_paths:
            archive.write(audio, audio.name)
    return PitchExport(archive_path, manifest_path, tuple(audio_paths), steps)
