"""Multilingual XLS-R embedding extraction for speech emotion recognition."""

from typing import Protocol

import numpy as np
import pandas as pd


XLSR_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
XLSR_SAMPLE_RATE = 16000


class FeatureExtractionError(ValueError):
    """Raised when audio cannot produce one finite feature vector."""


class Embedder(Protocol):
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


class XlsrEmbedder:
    """Lazy XLS-R loader that uses Apple Silicon acceleration when available."""

    def __init__(self, model_id: str = XLSR_MODEL_ID) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != XLSR_SAMPLE_RATE:
            raise FeatureExtractionError("XLS-R requires 16 kHz audio")

        import torch

        inputs = self.processor(samples, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            hidden_states = self.model(**inputs).last_hidden_state
        return hidden_states.mean(dim=1).squeeze(0).detach().cpu().numpy()


_default_embedder: XlsrEmbedder | None = None


def _get_default_embedder() -> XlsrEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = XlsrEmbedder()
    return _default_embedder


def extract_xlsr_embedding(
    samples: np.ndarray, sample_rate: int, embedder: Embedder | None = None
) -> pd.Series:
    """Return one named, finite XLS-R embedding vector for an utterance."""
    try:
        vector = (embedder or _get_default_embedder()).embed(samples, sample_rate)
    except FeatureExtractionError:
        raise
    except Exception as error:
        raise FeatureExtractionError("could not extract XLS-R features") from error

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise FeatureExtractionError("feature vector contains non-finite values")
    return pd.Series(vector, index=[f"xlsr_{index:04d}" for index in range(vector.size)])
