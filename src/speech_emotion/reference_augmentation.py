"""Traceable audio experiments inspired by the supplied SER slide deck."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import librosa
import numpy as np
import soundfile as sf

from speech_emotion.audio import AudioValidationError, load_valid_audio


METHODS = {
    "rotation": "時間旋轉",
    "gain": "音量調整",
    "rate": "變速調頻",
    "pitch": "固定時長變調",
}


@dataclass(frozen=True)
class ReferenceExport:
    archive: Path
    original_preview: Path
    outputs: tuple[Path, ...]
    rows: list[list]


def validate_parameters(methods, rotation, gain, rate, pitch):
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError("請選擇至少一種擴增方法。")
    for key, value, low, high, neutral in (
        ("rotation", rotation, 1, 99, None),
        ("gain", gain, 0.25, 1.5, 1),
        ("rate", rate, 0.5, 1.5, 1),
        ("pitch", pitch, -6, 6, 0),
    ):
        if key not in methods:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("調整參數必須是數字。")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError("調整參數超出範圍。")
        if value == neutral:
            raise ValueError(f"{METHODS[key]}的參數沒有改變聲音，請調整數值。")


def transform(samples, sr, method, value):
    """Apply exactly one operation, always from the original waveform."""
    if method == "rotation":
        cut = round(len(samples) * value / 100)
        result = np.concatenate((samples[cut:], samples[:cut]))
    elif method == "gain":
        result = samples * value
    elif method == "rate":
        result = librosa.resample(samples, orig_sr=sr, target_sr=sr / value,
                                  res_type="soxr_hq")
    elif method == "pitch":
        result = librosa.effects.pitch_shift(samples, sr=sr, n_steps=value,
                                              res_type="soxr_hq")
    else:
        raise ValueError("不支援此擴增方法。")
    result = np.asarray(result, dtype=np.float32)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("處理結果含有無效音訊。")
    peak = float(np.max(np.abs(result)))
    limiter = min(1.0, 0.99 / peak) if peak else 1.0
    return result * limiter, limiter


def export_reference_batch(sources, output_root, methods=("rotation",),
                           rotation=40, gain=0.75, rate=0.75, pitch=-4):
    validate_parameters(methods, rotation, gain, rate, pitch)
    if not sources:
        raise ValueError("請先錄音或上傳音檔。")
    if len(sources) > 100:
        raise ValueError("一次最多處理 100 個音檔。")
    output_root = Path(output_root).resolve()
    prepared, seen = [], set()
    total_samples = 0
    for path in sources:
        path = Path(path).resolve()
        if not path.is_file() or path.is_relative_to(output_root):
            raise ValueError("請選取原始音檔，不能重複擴增本工具的輸出。")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        try:
            samples, sr = load_valid_audio(path)
        except AudioValidationError as error:
            raise ValueError("音檔無法讀取、過短或為靜音，請重新選取。") from error
        if not np.isfinite(samples).all():
            raise ValueError("來源音訊含有無效數值。")
        total_samples += len(samples)
        if total_samples > 16000 * 60 * 30:
            raise ValueError("一次最多處理 30 分鐘的音訊。")
        prepared.append((path, digest, samples, sr))
        seen.add(digest)
    directory = output_root / uuid4().hex
    directory.mkdir(parents=True)
    parameters = dict(rotation=rotation, gain=gain, rate=rate, pitch=pitch)
    outputs, entries, rows = [], [], []
    try:
        (directory / "originals").mkdir()
        for index, (path, digest, samples, sr) in enumerate(prepared):
            backup = directory / "originals" / f"{digest}{path.suffix}"
            shutil.copy2(path, backup)
            if index == 0:
                preview = directory / "original-preview.wav"
                sf.write(preview, samples, sr, subtype="PCM_16")
            for method in dict.fromkeys(methods):
                result, limiter = transform(samples, sr, method, parameters[method])
                name = f"{digest[:16]}__{method}_{parameters[method]:g}.wav"
                target = directory / name
                sf.write(target, result, sr, subtype="PCM_16")
                outputs.append(target)
                entries.append(dict(source_name=path.name, group_id=digest,
                    source_sha256=digest, source_backup=backup.relative_to(directory).as_posix(),
                    output_name=name, method=method, parameter=parameters[method],
                    sample_rate=sr, samples=len(result), peak_limiter_gain=limiter,
                    identity_preserved="not_verified", emotion_preserved="not_verified",
                    split="unassigned"))
                rows.append([path.name, METHODS[method], parameters[method],
                             round(len(samples) / sr, 3), round(len(result) / sr, 3)])
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(dict(
            reference="華語語音情緒辨識(1).pptx, slides 23-25",
            fidelity="Reference-inspired implementation, not an exact research reproduction",
            automatically_enrolled_or_trained=False, originals_modified=False,
            split_policy="Split originals first; augment training only; never split one group_id across sets.",
            items=entries), ensure_ascii=False, indent=2), encoding="utf-8")
        archive = directory / "audio-experiments.zip"
        with ZipFile(archive, "x", ZIP_DEFLATED) as package:
            package.write(manifest, manifest.name)
            for target in outputs:
                package.write(target, target.name)
    except Exception:
        # Only remove this freshly created UUID batch, never any source files.
        shutil.rmtree(directory)
        raise
    return ReferenceExport(archive, preview, tuple(outputs), rows)
