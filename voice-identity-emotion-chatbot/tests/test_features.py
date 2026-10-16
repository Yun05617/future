import numpy as np

from speech_emotion.features import extract_xlsr_embedding


class FakeEmbedder:
    def embed(self, samples, sample_rate):
        assert sample_rate == 16000
        return np.array([0.25, -0.5, 1.0], dtype=np.float32)


def test_returns_finite_named_xlsr_features():
    samples = np.sin(np.linspace(0, 200, 32000, dtype=np.float32))
    features = extract_xlsr_embedding(samples, 16000, embedder=FakeEmbedder())
    assert list(features.index) == ["xlsr_0000", "xlsr_0001", "xlsr_0002"]
    assert features.index.is_unique
    assert np.isfinite(features.to_numpy()).all()
