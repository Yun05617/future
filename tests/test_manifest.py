import pandas as pd
import pytest
from pathlib import Path

from speech_emotion.manifest import ManifestValidationError, load_manifest


def test_rejects_speaker_in_multiple_splits(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame(
        [
            {"path": "a.wav", "speaker_id": "S01", "language": "zh", "emotion": "happy", "split": "train"},
            {"path": "b.wav", "speaker_id": "S01", "language": "zh", "emotion": "sad", "split": "test"},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ManifestValidationError, match="multiple splits"):
        load_manifest(path)


def test_rejects_blank_speaker_id(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame(
        [
            {"path": "a.wav", "speaker_id": " ", "language": "zh", "emotion": "happy", "split": "train"},
            {"path": "b.wav", "speaker_id": "", "language": "zh", "emotion": "sad", "split": "validation"},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ManifestValidationError, match="missing or blank"):
        load_manifest(path)


def test_provenance_has_required_columns():
    text = Path("reports/provenance.md").read_text(encoding="utf-8")
    assert "Source URL" in text
    assert "Licence" in text
    assert "Decision" in text
