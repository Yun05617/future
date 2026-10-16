from pathlib import Path

import numpy as np
import pytest

from speech_emotion.speaker_identity import (
    KNOWN_SPEAKERS,
    UNKNOWN_SPEAKER,
    enroll_speaker,
    identify_speaker,
    rebuild_speaker_profiles,
)
from speech_emotion.workspace import ProjectWorkspace


class FakeEmbedder:
    def embed(self, path: Path) -> np.ndarray:
        if "莊媗丞" in path.parts:
            return np.array([1.0, 0.0])
        if "姜昀廷" in path.parts:
            return np.array([0.0, 1.0])
        vectors = {
            "new.wav": np.array([0.95, 0.05]),
            "unknown.wav": np.array([0.71, 0.70]),
        }
        return vectors[path.name]


def _enrolled_workspace(tmp_path, wav_factory) -> tuple[ProjectWorkspace, Path, Path]:
    workspace = ProjectWorkspace(tmp_path / "專題研究")
    positive = wav_factory("positive.wav", np.ones(16_000, dtype=np.float32))
    negative = wav_factory("negative.wav", np.ones(16_000, dtype=np.float32))
    enroll_speaker(positive, "莊媗丞", workspace)
    enroll_speaker(negative, "姜昀廷", workspace)
    rebuild_speaker_profiles(workspace, FakeEmbedder())
    return workspace, positive, negative


def test_identify_returns_known_speaker_with_highest_similarity(tmp_path, wav_factory):
    workspace, _, _ = _enrolled_workspace(tmp_path, wav_factory)
    result = identify_speaker(
        Path("new.wav"), workspace, FakeEmbedder(), threshold=0.70, margin=0.05
    )

    assert result.speaker == "莊媗丞"
    assert result.is_unknown is False
    assert set(result.scores) == set(KNOWN_SPEAKERS)


def test_identify_returns_unknown_for_near_tied_scores(tmp_path, wav_factory):
    workspace, _, _ = _enrolled_workspace(tmp_path, wav_factory)
    result = identify_speaker(
        Path("unknown.wav"), workspace, FakeEmbedder(), threshold=0.70, margin=0.05
    )

    assert result.speaker == UNKNOWN_SPEAKER
    assert result.is_unknown is True


@pytest.mark.parametrize("score, expected_unknown", [(0.3999, True), (0.40, True), (0.4001, False), (0.5857, False)])
def test_local_policy_accepts_only_scores_above_forty_percent(tmp_path, monkeypatch, score, expected_unknown):
    from speech_emotion import personal_demo_ui

    workspace = ProjectWorkspace(tmp_path / "profiles")
    centroid = np.array([score, np.sqrt(1 - score**2)], dtype=np.float32)
    np.save(workspace.root / "03_語音身分辨識/成員聲紋範本/姜昀廷.npy", centroid)

    class Embedder:
        def embed(self, path):
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(personal_demo_ui, "WORKSPACE", workspace)
    identity = personal_demo_ui._LocalIdentity()
    identity.embedder = Embedder()
    result = identity.identify(Path("input.wav"))

    assert result.is_unknown is expected_unknown
    assert result.speaker == (UNKNOWN_SPEAKER if expected_unknown else "姜昀廷")


def test_local_policy_does_not_require_three_percentage_point_gap(tmp_path, monkeypatch):
    from speech_emotion import personal_demo_ui

    workspace = ProjectWorkspace(tmp_path / "profiles")
    for name, score in [("姜昀廷", 0.45), ("趙科翰", 0.449)]:
        np.save(
            workspace.root / "03_語音身分辨識/成員聲紋範本" / f"{name}.npy",
            np.array([score, np.sqrt(1 - score**2)], dtype=np.float32),
        )

    class Embedder:
        def embed(self, path):
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(personal_demo_ui, "WORKSPACE", workspace)
    identity = personal_demo_ui._LocalIdentity()
    identity.embedder = Embedder()
    assert identity.identify(Path("input.wav")).speaker == "姜昀廷"


def test_local_policy_keeps_unknown_when_no_profiles_exist(tmp_path, monkeypatch):
    from speech_emotion import personal_demo_ui

    monkeypatch.setattr(personal_demo_ui, "WORKSPACE", ProjectWorkspace(tmp_path / "profiles"))
    identity = personal_demo_ui._LocalIdentity()
    identity.embedder = FakeEmbedder()
    assert identity.identify(Path("new.wav")).speaker == UNKNOWN_SPEAKER
