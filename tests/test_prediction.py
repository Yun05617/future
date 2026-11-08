import joblib
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from speech_emotion.prediction import predict_file


@pytest.fixture
def saved_model(tmp_path):
    pipeline = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(random_state=42), cv=2, ensemble=False),
    )
    pipeline.fit([[0.0], [1.0], [0.1], [0.9]], ["happy", "sad", "happy", "sad"])
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": ["f1"],
            "labels": ["happy", "sad"],
            "model_version": "test",
        },
        path,
    )
    return path


def test_prediction_returns_complete_contract(monkeypatch, tmp_path, saved_model):
    monkeypatch.setattr(
        "speech_emotion.prediction.features_for_path", lambda _: pd.Series({"f1": 0.2})
    )
    result = predict_file(tmp_path / "voice.wav", saved_model, "u-001")
    assert result["utterance_id"] == "u-001"
    assert result["status"] == "ok"
    assert result["speech_emotion"] in {"neutral", "happy", "angry", "sad"}
    assert result["model_version"] == "test"
    assert set(result["scores"]) == {"neutral", "happy", "angry", "sad"}
    assert result["latency_ms"] >= 0
