from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def wav_factory(tmp_path: Path):
    def create(name: str, samples: np.ndarray, sample_rate: int = 16_000) -> Path:
        path = tmp_path / name
        sf.write(path, samples, sample_rate)
        return path

    return create
