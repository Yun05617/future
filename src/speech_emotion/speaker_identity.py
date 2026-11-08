"""Local speaker enrollment and identity matching for the four project members."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol
import shutil
import warnings
from uuid import uuid4

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from speech_emotion.audio import load_valid_audio
from speech_emotion.workspace import ProjectWorkspace, WorkspaceError


KNOWN_SPEAKERS = ("莊媗丞", "姜昀廷", "紀律廷", "趙科翰")
UNKNOWN_SPEAKER = "未知使用者"
IDENTITY_THRESHOLD = 0.40
IDENTITY_MARGIN = 0.0
IDENTITY_MIN_SUPPORT = 0.80


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
    classifier_scores: dict[str, float] = field(default_factory=dict)
    decision_method: str = "centroid"
    model_version: str | None = None
    support_threshold: float | None = None


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
    """Re-extract embeddings and fit a balanced identity head, then publish atomically."""
    root = workspace.root / "03_語音身分辨識"
    lock = root / "identity-update.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise SpeakerIdentityError("身分模型正在更新，請等待完成；若上次程式異常關閉，請先確認沒有訓練程序再移除 identity-update.lock。") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        return _fit_speaker_profiles(workspace, embedder)
    finally:
        lock.unlink(missing_ok=True)


def _fit_speaker_profiles(workspace, embedder):
    counts: dict[str, int] = {}
    unique_counts, prototypes, vectors_by_speaker = {}, {}, {}
    dimension = None
    cache_hits = 0
    quality_warnings = []
    cache_key = getattr(embedder, "cache_key", None)
    cache_directory = workspace.root / "03_語音身分辨識" / "特徵快取"
    if cache_key:
        cache_directory.mkdir(exist_ok=True)
    for speaker in KNOWN_SPEAKERS:
        recordings = sorted(_enrollment_directory(workspace, speaker).glob("*.wav"))
        counts[speaker] = len(recordings)
        vectors, seen = [], set()
        for recording in recordings:
            digest = hashlib.sha256(recording.read_bytes()).hexdigest()
            if digest in seen:
                continue
            diagnose = getattr(embedder, "diagnose", None)
            if callable(diagnose):
                quality = diagnose(recording)
                if quality.get("flags"):
                    quality_warnings.append(dict(speaker=speaker, file=recording.name, **quality))
            vector = None
            cache_path = None
            if cache_key:
                key = hashlib.sha256(f"{cache_key}:{digest}".encode()).hexdigest()
                cache_path = cache_directory / f"{key}.npy"
                if cache_path.exists():
                    try:
                        cached = np.load(cache_path, allow_pickle=False)
                        if cached.ndim == 1 and len(cached):
                            vector = _unit(cached)
                            cache_hits += 1
                    except (OSError, ValueError, TypeError):
                        pass
            if vector is None:
                vector = _unit(embedder.embed(recording))
                if cache_path is not None:
                    temporary_cache = cache_directory / f".{uuid4().hex}.npy"
                    try:
                        np.save(temporary_cache, vector)
                        os.replace(temporary_cache, cache_path)
                    finally:
                        temporary_cache.unlink(missing_ok=True)
            if dimension is not None and dimension != len(vector):
                raise SpeakerIdentityError("聲紋特徵維度不一致；保留上次模型。")
            dimension = len(vector)
            vectors.append(vector)
            seen.add(digest)
        unique_counts[speaker] = len(vectors)
        if vectors:
            prototypes[speaker] = _unit(np.mean(vectors, axis=0))
            vectors_by_speaker[speaker] = vectors

    eligible = [s for s in KNOWN_SPEAKERS if unique_counts[s] >= 2]
    classifier = None
    if len(eligible) >= 2:
        features = np.asarray([v for s in eligible for v in vectors_by_speaker[s]])
        labels = [s for s in eligible for _ in vectors_by_speaker[s]]
        classifier = LogisticRegression(C=10.0, class_weight="balanced", solver="lbfgs",
                                        max_iter=2000, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(features, labels)

    version = uuid4().hex
    metadata = dict(schema_version=1, version=version,
        trained_at=datetime.now(timezone.utc).isoformat(), counts=counts,
        unique_counts=unique_counts, trained_classes=list(classifier.classes_) if classifier else [],
        mode="logistic_regression_with_cosine_gate" if classifier else "centroid_only",
        base_model_updated=False, embedder_source=getattr(embedder, "source", type(embedder).__name__),
        minimum_unique_recordings_per_class=2, class_weight="balanced",
        validation_accuracy=None, probabilities_calibrated=False,
        feature_cache_hits=cache_hits,
        feature_cache_key=cache_key,
        newly_extracted_embeddings=sum(unique_counts.values())-cache_hits,
        quality_warnings=quality_warnings,
        note="No held-out evaluation; augmented copies are not independent real recordings.")
    names = list(prototypes)
    root = workspace.root / "03_語音身分辨識"
    target = root / "learned_identity.npz"
    temporary = root / f".{version}.npz"
    try:
        np.savez_compressed(temporary,
            metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
            names=np.array(names, dtype=str),
            centroids=np.asarray([prototypes[s] for s in names], dtype=np.float32).reshape(len(names), dimension or 0),
            classes=np.array(list(classifier.classes_) if classifier else [], dtype=str),
            coefficients=classifier.coef_ if classifier else np.empty((0, dimension or 0)),
            intercept=classifier.intercept_ if classifier else np.empty(0))
        if target.exists():
            history = root / "模型更新備份"
            history.mkdir(exist_ok=True)
            shutil.copy2(target, history / f"before-{version}.npz")
        # One file contains centroids and learned weights so readers never see a partial update.
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return counts


def _load_learned_profiles(workspace):
    path = workspace.root / "03_語音身分辨識" / "learned_identity.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            snapshot = {key: data[key].copy() for key in
                        ("names", "centroids", "classes", "coefficients", "intercept")}
            snapshot["metadata"] = json.loads(str(data["metadata"].item()))
        names, centroids = snapshot["names"].tolist(), snapshot["centroids"]
        classes, coefficients, intercept = snapshot["classes"].tolist(), snapshot["coefficients"], snapshot["intercept"]
        if (snapshot["metadata"]["schema_version"] != 1 or
            any(s not in KNOWN_SPEAKERS for s in names) or len(set(names)) != len(names) or
            centroids.ndim != 2 or len(centroids) != len(names) or
            any(s not in names for s in classes) or len(set(classes)) != len(classes) or
            (classes and (len(classes) < 2 or coefficients.shape != (1 if len(classes) == 2 else len(classes), centroids.shape[1]) or
                          intercept.shape != (len(coefficients),))) or
            not all(np.isfinite(a).all() for a in (centroids, coefficients, intercept))):
            raise ValueError("invalid identity snapshot")
        return snapshot
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SpeakerIdentityError("學習後的身分模型無法讀取，請重新更新；未改用損壞模型。") from error


def speaker_training_summary(workspace):
    snapshot = _load_learned_profiles(workspace)
    return snapshot["metadata"] if snapshot else None


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
    mode: str = "hybrid",
    min_support: float = 0.0,
) -> IdentityResult:
    """Select with a classifier alone or the legacy hybrid cosine-gated policy."""
    started = perf_counter()
    if mode not in {"hybrid", "classifier_only"}:
        raise SpeakerIdentityError("不支援的身分辨識模式。")
    if not np.isfinite(min_support) or not 0.0 <= min_support <= 1.0:
        raise SpeakerIdentityError("分類器支持度門檻必須介於 0 與 1。")
    if not 0.0 <= threshold <= 1.0 or not np.isfinite(margin) or margin < 0.0:
        raise SpeakerIdentityError("身分辨識門檻設定不正確")
    vector = _unit(embedder.embed(audio_path))
    snapshot = _load_learned_profiles(workspace)
    if snapshot is not None:
        previous_key = snapshot['metadata'].get('feature_cache_key')
        if previous_key and previous_key != getattr(embedder, 'cache_key', None):
            raise SpeakerIdentityError("底層聲紋模型版本已改變，請重新更新身分模型。")
    profiles = dict(zip(snapshot["names"].tolist(), snapshot["centroids"])) if snapshot else None
    scores: dict[str, float] = {}
    for speaker in KNOWN_SPEAKERS:
        centroid = profiles.get(speaker) if profiles is not None else _load_centroid(workspace, speaker)
        if centroid is not None and centroid.shape != vector.shape:
            raise SpeakerIdentityError("聲紋模型維度已改變，請重新更新身分模型。")
        scores[speaker] = float(np.dot(vector, centroid)) if centroid is not None else -1.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_score = ranked[0]
    second_score = ranked[1][1]
    classifier_scores = {}
    decision_method = "classifier_unavailable" if mode == "classifier_only" else "centroid"
    use_classifier = snapshot is not None and len(snapshot["classes"]) >= 2 and (
        mode == "classifier_only" or best_speaker in snapshot["classes"]
    )
    if use_classifier:
        source = getattr(embedder, "source", type(embedder).__name__)
        if source != snapshot["metadata"]["embedder_source"]:
            raise SpeakerIdentityError("聲紋特徵模型已改變，請重新訓練身分分類器。")
        logits = snapshot["coefficients"] @ vector + snapshot["intercept"]
        if len(snapshot["classes"]) == 2:
            logits = np.array([0.0, logits[0]])
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        classifier_scores = dict(zip(snapshot["classes"].tolist(), map(float, probabilities)))
        best_speaker = max(classifier_scores, key=classifier_scores.get)
        best_score = scores[best_speaker]
        second_score = max(v for s, v in scores.items() if s != best_speaker)
        decision_method = "classifier_only" if mode == "classifier_only" else "learned_classifier"
    # Scores are float32; compare at the same precision to keep 40% exclusive.
    if mode == "classifier_only":
        # Closed-set mode: raw similarities are diagnostics, never acceptance gates.
        is_unknown = not classifier_scores or max(classifier_scores.values()) < min_support
    else:
        is_unknown = best_score <= float(np.float32(threshold)) or (margin > 0 and best_score - second_score < margin)
    return IdentityResult(
        speaker=UNKNOWN_SPEAKER if is_unknown else best_speaker,
        scores=scores,
        is_unknown=is_unknown,
        latency_ms=float((perf_counter() - started) * 1000),
        classifier_scores=classifier_scores,
        decision_method=decision_method,
        model_version=snapshot["metadata"]["version"] if snapshot else None,
        support_threshold=min_support if mode == "classifier_only" else None,
    )


class SpeechBrainEmbedder:
    """Lazy SpeechBrain adapter so imports and tests do not download model weights."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        self.source = source
        self._classifier = None

    @property
    def cache_key(self):
        # A pinned local HF revision prevents mixing embeddings across model updates.
        cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", Path.home()/'.cache/huggingface/hub'))
        ref = cache / ('models--' + self.source.replace('/', '--')) / 'refs' / 'main'
        try:
            revision = ref.read_text(encoding='utf-8').strip()
        except OSError:
            return None
        if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            return None
        return f'{self.source}:{revision}:encode-batch-legacy-v1'

    def diagnose(self, path):
        from speech_emotion.identity_diagnostics import signal_diagnostics
        samples, rate = load_valid_audio(path)
        return signal_diagnostics(samples, rate)

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
        with torch.inference_mode():
            embedding = self._classifier.encode_batch(waveform)
        return np.asarray(embedding.squeeze().detach().cpu().numpy(), dtype=np.float32)
