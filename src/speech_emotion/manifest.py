"""Validation for speech-emotion dataset manifests."""

from pathlib import Path

import pandas as pd


class ManifestValidationError(ValueError):
    """Raised when a dataset manifest is incomplete or unsafe to use."""


REQUIRED_COLUMNS = {"path", "speaker_id", "language", "emotion", "split"}
ALLOWED_EMOTIONS = {"neutral", "happy", "angry", "sad"}
ALLOWED_SPLITS = {"train", "validation", "test"}


def load_manifest(path: Path) -> pd.DataFrame:
    """Load a manifest and ensure speakers never cross data splits."""
    try:
        manifest = pd.read_csv(path)
    except Exception as error:
        raise ManifestValidationError(f"unable to read manifest: {path}") from error

    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ManifestValidationError(f"missing required columns: {sorted(missing)}")
    required_values = manifest[list(REQUIRED_COLUMNS)]
    if required_values.isna().any().any() or required_values.astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ManifestValidationError("manifest contains missing or blank required values")
    if manifest["path"].duplicated().any():
        raise ManifestValidationError("manifest contains duplicate paths")
    if not manifest["emotion"].isin(ALLOWED_EMOTIONS).all():
        raise ManifestValidationError("manifest contains an unsupported emotion")
    if not manifest["split"].isin(ALLOWED_SPLITS).all():
        raise ManifestValidationError("manifest contains an unsupported split")

    split_counts = manifest.groupby("speaker_id")["split"].nunique()
    if (split_counts > 1).any():
        raise ManifestValidationError("a speaker appears in multiple splits")

    return manifest
