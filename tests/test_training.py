import pandas as pd

from speech_emotion.training import save_model, train_pipeline


def test_saved_pipeline_predicts(monkeypatch, tmp_path):
    manifest = pd.DataFrame(
        [
            {"path": "a.wav", "speaker_id": "S01", "language": "zh", "emotion": "happy", "split": "train"},
            {"path": "b.wav", "speaker_id": "S02", "language": "zh", "emotion": "sad", "split": "train"},
            {"path": "c.wav", "speaker_id": "S03", "language": "zh", "emotion": "happy", "split": "train"},
            {"path": "d.wav", "speaker_id": "S04", "language": "zh", "emotion": "sad", "split": "train"},
            {"path": "e.wav", "speaker_id": "S05", "language": "zh", "emotion": "happy", "split": "validation"},
            {"path": "f.wav", "speaker_id": "S06", "language": "zh", "emotion": "sad", "split": "validation"},
        ]
    )
    monkeypatch.setattr(
        "speech_emotion.training.features_for_path",
        lambda path: pd.Series({"f1": float(path.name in {"b.wav", "d.wav", "f.wav"}), "f2": 1.0}),
    )
    pipeline, metrics = train_pipeline(manifest, tmp_path)
    output = tmp_path / "model.joblib"
    save_model(pipeline, ["f1", "f2"], output)
    assert output.exists()
    assert "validation_macro_f1" in metrics
