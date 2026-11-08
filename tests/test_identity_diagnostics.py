import numpy as np
import pytest

from speech_emotion.identity_diagnostics import signal_diagnostics, trim_identity_edges


def test_diagnostics_flags_silence_and_short_audio():
    result = signal_diagnostics(np.zeros(16000,dtype=np.float32),16000)
    assert result['relative_active_s']==0
    assert {'low_volume','short_recording','high_relative_silence'} <= set(result['flags'])


def test_clipping_detected_without_modifying_source():
    y=np.ones(32000,dtype=np.float32)
    before=y.copy()
    assert 'clipping' in signal_diagnostics(y,16000)['flags']
    np.testing.assert_array_equal(y,before)


def test_edge_trim_retains_internal_pause_and_context():
    sr=16000
    y=np.r_[np.zeros(sr),np.ones(sr)*0.1,np.zeros(sr//2),np.ones(sr)*0.1,np.zeros(sr)].astype(np.float32)
    z=trim_identity_edges(y,sr)
    assert len(z)<len(y)
    assert len(z)>=sr*2.5
    assert np.count_nonzero(z)==sr*2
    assert len(y)==sr*4.5


@pytest.mark.parametrize('y',[np.array([]),np.array([np.nan]),np.zeros((2,3))])
def test_invalid_audio_rejected(y):
    with pytest.raises(ValueError):
        signal_diagnostics(y,16000)
