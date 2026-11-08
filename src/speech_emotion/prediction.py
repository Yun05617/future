"""Single-file inference for the speech-emotion baseline."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from speech_emotion.training import features_for_path


EMOTIONS = ("neutral", "happy", "angry", "sad")


def predict_file(audio_path: Path, model_path: Path, utterance_id: str) -> dict[str, object]:
    """Classify one recording and always return the public result contract."""
    started = perf_counter()
    try:
        artifact = joblib.load(model_path)
        features = features_for_path(audio_path)
        ordered = np.asarray([features[name] for name in artifact["feature_names"]]).reshape(1, -1)
        pipeline = artifact["pipeline"]
        probabilities = pipeline.predict_proba(ordered)[0]
        class_scores = dict(zip(pipeline.classes_, probabilities, strict=True))
        scores = {emotion: float(class_scores.get(emotion, 0.0)) for emotion in EMOTIONS}
        label = max(scores, key=scores.get)
        return {
            "utterance_id": utterance_id,
            "status": "ok",
            "speech_emotion": label,
            "scores": scores,
            "latency_ms": (perf_counter() - started) * 1000,
            "model_version": artifact["model_version"],
        }
    except Exception:
        return {
            "utterance_id": utterance_id,
            "status": "error",
            "message": "這段錄音無法分析，請在安靜環境重新錄製一段至少半秒的語音。",
            "latency_ms": (perf_counter() - started) * 1000,
        }
