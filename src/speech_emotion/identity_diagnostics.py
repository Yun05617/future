"""Non-destructive signal diagnostics and experimental edge-silence trimming."""

import numpy as np
import librosa


def signal_diagnostics(samples, sr):
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.isfinite(sr) or sr <= 0:
        raise ValueError('音訊必須是一維、非空且包含有效取樣值。')
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    intervals = librosa.effects.split(y, top_db=30, frame_length=512, hop_length=160)
    active = int(sum(end - start for start, end in intervals)) if peak > 1e-4 else 0
    flags = []
    if len(y) / sr < 2:
        flags.append('short_recording')
    if rms < 0.005:
        flags.append('low_volume')
    clipped = float(np.mean(np.abs(y) >= 0.999))
    if clipped > 0.001:
        flags.append('clipping')
    if active / len(y) < 0.5:
        flags.append('high_relative_silence')
    return dict(duration_s=len(y)/sr, rms=rms, peak=peak,
                clipping_ratio=clipped, relative_active_s=active/sr,
                relative_active_ratio=active/len(y), flags=flags)


def trim_identity_edges(samples, sr):
    """Energy-based experiment, not a speech/voice activity detector."""
    y = np.asarray(samples, dtype=np.float32)
    intervals = librosa.effects.split(y, top_db=30, frame_length=512, hop_length=160)
    if not len(intervals) or np.max(np.abs(y)) < 1e-4:
        return y.copy()
    pad = round(sr * 0.2)
    start, end = max(0, intervals[0, 0]-pad), min(len(y), intervals[-1, 1]+pad)
    return y[start:end].copy() if end-start >= sr else y.copy()
