"""Smoke checks for the local speech-emotion training environment."""


def test_required_training_packages_import() -> None:
    import joblib  # noqa: F401
    import librosa  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import soundfile  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
