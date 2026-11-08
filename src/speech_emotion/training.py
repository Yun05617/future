"""Training and persistence for the speech-emotion baseline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from speech_emotion.audio import load_valid_audio
from speech_emotion.features import extract_xlsr_embedding


MODEL_VERSION = "xlsr-300m-svc-v1"


def features_for_path(path: Path) -> pd.Series:
    """Load one recording and calculate its acoustic feature vector."""
    samples, sample_rate = load_valid_audio(path)
    return extract_xlsr_embedding(samples, sample_rate)


def train_pipeline(manifest: pd.DataFrame, audio_root: Path) -> tuple[Pipeline, dict[str, float]]:
    """Train on train rows and measure macro F1 on validation rows."""
    feature_rows = [features_for_path(audio_root / path) for path in manifest["path"]]
    feature_table = pd.DataFrame(feature_rows).reset_index(drop=True)
    labels = manifest["emotion"].reset_index(drop=True)
    splits = manifest["split"].reset_index(drop=True)

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    if train_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError("manifest needs both train and validation rows")
    if labels[train_mask].nunique() < 2:
        raise ValueError("training data needs at least two emotion classes")

    pipeline = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=42),
            cv=2,
            ensemble=False,
        ),
    )
    pipeline.fit(feature_table.loc[train_mask], labels.loc[train_mask])
    validation_predictions = pipeline.predict(feature_table.loc[validation_mask])
    metrics = {
        "validation_macro_f1": float(
            f1_score(labels.loc[validation_mask], validation_predictions, average="macro", zero_division=0)
        )
    }
    return pipeline, metrics


def save_model(pipeline: Pipeline, feature_names: list[str], destination: Path) -> None:
    """Save the fitted pipeline and metadata needed by inference."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": feature_names,
            "labels": list(pipeline.classes_),
            "model_version": MODEL_VERSION,
        },
        destination,
    )
