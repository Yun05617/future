"""Storage helpers for the local three-emotion personal demo."""

import json
from pathlib import Path
import shutil
from time import perf_counter
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from speech_emotion.audio import AudioValidationError, normalize_audio_to_wav
from speech_emotion.features import XLSR_MODEL_ID
from speech_emotion.training import features_for_path


PERSONAL_LABELS = ("positive", "negative", "calm")


class PersonalDemoError(ValueError):
    """Raised when a personal-demo recording cannot be stored."""


def training_counts(recordings_root: Path) -> dict[str, int]:
    """Return the number of stored WAV recordings for each personal label."""
    return {
        label: len(list((recordings_root / label).glob("*.wav")))
        for label in PERSONAL_LABELS
    }


def add_training_audio(
    source_path: Path, label: str, recordings_root: Path
) -> dict[str, int]:
    """Validate and copy one local recording, then return all label counts."""
    _store_training_audio(source_path, label, recordings_root)
    return training_counts(recordings_root)


def _store_training_audio(source_path: Path, label: str, recordings_root: Path) -> Path:
    """Validate and copy one recording, returning its new private path."""
    if label not in PERSONAL_LABELS:
        raise PersonalDemoError("unsupported personal-demo label")

    destination = recordings_root / label / f"{uuid4().hex}.wav"
    try:
        normalize_audio_to_wav(source_path, destination)
    except AudioValidationError as error:
        raise PersonalDemoError("錄音檔無法使用，請重新錄製。") from error
    return destination


def _split_paths(
    paths_by_label: dict[str, list[Path]],
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """Create a deterministic within-speaker holdout for every label."""
    train_paths, train_labels, test_paths, test_labels = [], [], [], []
    for label, paths in paths_by_label.items():
        ordered = sorted(paths)
        holdout_count = max(1, round(len(ordered) / 3))
        holdout_count = min(holdout_count, len(ordered) - 1)
        test_group, train_group = ordered[:holdout_count], ordered[holdout_count:]
        train_paths.extend(train_group)
        train_labels.extend([label] * len(train_group))
        test_paths.extend(test_group)
        test_labels.extend([label] * len(test_group))
    return train_paths, train_labels, test_paths, test_labels


def _training_metadata_path(recordings_root: Path) -> Path:
    """Return the ignored generated metadata path for personal-demo training."""
    return recordings_root.parents[1] / "processed" / "personal_demo" / "training_metadata.json"


def _feature_cache_path(recordings_root: Path) -> Path:
    """Return the local cache used to avoid repeating XLS-R inference."""
    return recordings_root.parents[1] / "processed" / "personal_demo" / "xlsr_features.joblib"


def _load_feature_cache(recordings_root: Path) -> dict[str, object]:
    """Load a cache only when it belongs to the current XLS-R model."""
    cache_path = _feature_cache_path(recordings_root)
    if not cache_path.exists():
        return {"model_id": XLSR_MODEL_ID, "items": {}}
    try:
        cache = joblib.load(cache_path)
    except Exception:
        return {"model_id": XLSR_MODEL_ID, "items": {}}
    if not isinstance(cache, dict) or cache.get("model_id") != XLSR_MODEL_ID:
        return {"model_id": XLSR_MODEL_ID, "items": {}}
    items = cache.get("items")
    if not isinstance(items, dict):
        return {"model_id": XLSR_MODEL_ID, "items": {}}
    return {"model_id": XLSR_MODEL_ID, "items": items}


def _save_feature_cache(recordings_root: Path, cache: dict[str, object]) -> None:
    """Persist extracted vectors beside the generated personal-demo model."""
    cache_path = _feature_cache_path(recordings_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, cache_path, compress=3)


def _features_for_paths(
    paths: list[Path], recordings_root: Path
) -> dict[Path, pd.Series]:
    """Extract only new or modified recordings, reusing local XLS-R vectors."""
    cache = _load_feature_cache(recordings_root)
    items = cache["items"]
    assert isinstance(items, dict)
    features: dict[Path, pd.Series] = {}
    changed = False
    newly_extracted = 0
    for path in paths:
        stat = path.stat()
        key = str(path.resolve())
        cached = items.get(key)
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if isinstance(cached, dict) and cached.get("signature") == signature:
            values = cached.get("values")
            if isinstance(values, dict):
                features[path] = pd.Series(values, dtype="float32")
                continue
        features[path] = features_for_path(path)
        items[key] = {"signature": signature, "values": features[path].to_dict()}
        changed = True
        newly_extracted += 1
        # Checkpoint during long first-time training so an interruption does
        # not discard every vector computed before it.
        if newly_extracted % 8 == 0:
            _save_feature_cache(recordings_root, cache)
    if changed:
        _save_feature_cache(recordings_root, cache)
    return features


def _correction_history_path(recordings_root: Path) -> Path:
    """Return the local history file used only for correction undo."""
    return recordings_root.parents[1] / "processed" / "personal_demo" / "correction_history.json"


def _manual_addition_history_path(recordings_root: Path) -> Path:
    """Return the local history file used only for manual-addition undo."""
    return (
        recordings_root.parents[1]
        / "processed"
        / "personal_demo"
        / "manual_addition_history.json"
    )


def _load_correction_history(recordings_root: Path) -> list[dict[str, str]]:
    """Load validated local correction history, rejecting malformed data."""
    history_path = _correction_history_path(recordings_root)
    if not history_path.exists():
        return []
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersonalDemoError("更正紀錄無法讀取，無法安全撤回。") from error
    if not isinstance(history, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or item.get("label") not in PERSONAL_LABELS
        for item in history
    ):
        raise PersonalDemoError("更正紀錄格式不正確，無法安全撤回。")
    return history


def _save_correction_history(
    recordings_root: Path, history: list[dict[str, str]]
) -> None:
    """Persist the correction-only history alongside other generated metadata."""
    history_path = _correction_history_path(recordings_root)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def add_manual_training_audio(
    source_path: Path, label: str, recordings_root: Path
) -> dict[str, int]:
    """Store one top-of-page training recording and make it undoable."""
    history_path = _manual_addition_history_path(recordings_root)
    history: list[dict[str, str]] = []
    if history_path.exists():
        try:
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PersonalDemoError("加入紀錄無法讀取，無法安全儲存錄音。") from error
        if not isinstance(loaded_history, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or item.get("label") not in PERSONAL_LABELS
            for item in loaded_history
        ):
            raise PersonalDemoError("加入紀錄格式不正確，無法安全儲存錄音。")
        history = loaded_history
    destination = _store_training_audio(source_path, label, recordings_root)
    history.append({"path": str(destination), "label": label})
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return training_counts(recordings_root)


def undo_last_manual_training_audio(recordings_root: Path) -> dict[str, int]:
    """Remove only the newest recording stored through the manual-add UI."""
    history_path = _manual_addition_history_path(recordings_root)
    if not history_path.exists():
        raise PersonalDemoError("目前沒有可撤回的加入錄音。")
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersonalDemoError("加入紀錄無法讀取，無法安全撤回。") from error
    if not isinstance(history, list) or not history:
        raise PersonalDemoError("目前沒有可撤回的加入錄音。")
    last_entry = history[-1]
    if (
        not isinstance(last_entry, dict)
        or not isinstance(last_entry.get("path"), str)
        or last_entry.get("label") not in PERSONAL_LABELS
    ):
        raise PersonalDemoError("加入紀錄格式不正確，無法安全撤回。")

    label = last_entry["label"]
    target = Path(last_entry["path"]).resolve()
    allowed_directory = (recordings_root / label).resolve()
    try:
        target.relative_to(allowed_directory)
    except ValueError as error:
        raise PersonalDemoError("加入紀錄位置不正確，無法安全撤回。") from error
    if target.suffix.lower() != ".wav" or not target.is_file():
        raise PersonalDemoError("要撤回的加入錄音不存在。")
    target.unlink()
    history_path.write_text(
        json.dumps(history[:-1], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return training_counts(recordings_root)


def train_personal_demo(recordings_root: Path, model_path: Path) -> dict[str, object]:
    """Train a local three-class model and score a same-speaker holdout."""
    paths_by_label = {
        label: sorted((recordings_root / label).glob("*.wav"))
        for label in PERSONAL_LABELS
    }
    if any(len(paths) < 2 for paths in paths_by_label.values()):
        raise PersonalDemoError("每種情緒至少 2 段錄音才能訓練。")

    train_paths, train_labels, test_paths, test_labels = _split_paths(paths_by_label)
    feature_map = _features_for_paths(train_paths + test_paths, recordings_root)
    train_features = pd.DataFrame([feature_map[path] for path in train_paths])
    test_features = pd.DataFrame([feature_map[path] for path in test_paths])
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    pipeline.fit(train_features, train_labels)
    predictions = pipeline.predict(test_features)

    metrics: dict[str, object] = {
        "same_speaker_accuracy": float(accuracy_score(test_labels, predictions)),
        "same_speaker_macro_f1": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "same_speaker_train_count": len(train_paths),
        "same_speaker_test_count": len(test_paths),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "feature_names": list(train_features.columns),
        "labels": list(pipeline.classes_),
        "model_version": "personal-three-emotion-v1",
        "split_paths": {
            "train": [str(path) for path in train_paths],
            "test": [str(path) for path in test_paths],
        },
    }
    joblib.dump(artifact, model_path)

    metadata_path = _training_metadata_path(recordings_root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": artifact["model_version"],
                "labels": artifact["labels"],
                "split_paths": artifact["split_paths"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def correct_analysis_and_retrain(
    source_path: Path,
    corrected_label: str,
    recordings_root: Path,
    model_path: Path,
) -> dict[str, object]:
    """Store a corrected analysis recording and immediately retrain locally."""
    destination = _store_training_audio(source_path, corrected_label, recordings_root)
    history = _load_correction_history(recordings_root)
    history.append({"path": str(destination), "label": corrected_label})
    _save_correction_history(recordings_root, history)
    counts = training_counts(recordings_root)
    metrics = train_personal_demo(recordings_root, model_path)
    return {"counts": counts, **metrics}


def undo_last_correction_and_retrain(
    recordings_root: Path, model_path: Path
) -> dict[str, object]:
    """Undo only the newest correction recording and immediately retrain locally."""
    history = _load_correction_history(recordings_root)
    if not history:
        raise PersonalDemoError("目前沒有可撤回的更正紀錄。")

    last_correction = history[-1]
    label = last_correction["label"]
    target = Path(last_correction["path"]).resolve()
    allowed_directory = (recordings_root / label).resolve()
    try:
        target.relative_to(allowed_directory)
    except ValueError as error:
        raise PersonalDemoError("更正紀錄位置不正確，無法安全撤回。") from error
    if target.suffix.lower() != ".wav" or not target.is_file():
        raise PersonalDemoError("要撤回的更正錄音不存在。")

    original_bytes = target.read_bytes()
    target.unlink()
    _save_correction_history(recordings_root, history[:-1])
    try:
        metrics = train_personal_demo(recordings_root, model_path)
    except Exception:
        target.write_bytes(original_bytes)
        _save_correction_history(recordings_root, history)
        raise
    return {"counts": training_counts(recordings_root), **metrics}


def predict_personal_demo(audio_path: Path, model_path: Path) -> dict[str, object]:
    """Predict the three personal-demo emotions for one local recording."""
    started = perf_counter()
    try:
        artifact = joblib.load(model_path)
        pipeline = artifact["pipeline"]
        feature_names = artifact["feature_names"]
        features = features_for_path(audio_path)
        feature_frame = pd.DataFrame([features]).reindex(columns=feature_names)
        probabilities = pipeline.predict_proba(feature_frame)[0]
        classes = getattr(pipeline, "classes_", artifact.get("labels", ()))
        scores = {label: 0.0 for label in PERSONAL_LABELS}
        for label, probability in zip(classes, probabilities):
            if label in scores:
                scores[label] = float(probability)
        emotion = max(scores, key=scores.get)
        return {
            "status": "ok",
            "emotion": emotion,
            "scores": scores,
            "latency_ms": float((perf_counter() - started) * 1000),
            "model_version": artifact["model_version"],
        }
    except Exception:
        return {
            "status": "error",
            "message": "模型或錄音無法處理，請稍後再試。",
            "latency_ms": float((perf_counter() - started) * 1000),
        }
