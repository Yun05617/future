import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf

from speech_emotion.reference_augmentation import export_reference_batch, transform


def tone(path):
    sr = 16000
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr), sr)
    return path


def frequency(samples, sr):
    return np.argmax(np.abs(np.fft.rfft(samples * np.hanning(len(samples))))) * sr / len(samples)


def test_rotation_moves_exact_chunks_without_changing_samples():
    source = np.arange(100, dtype=np.float32) / 200
    rotated, gain = transform(source, 16000, 'rotation', 40)
    np.testing.assert_array_equal(rotated, np.r_[source[40:], source[:40]])
    assert gain == 1
    np.testing.assert_array_equal(np.sort(rotated), np.sort(source))


def test_gain_only_changes_amplitude():
    source = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    result, limiter = transform(source, 16000, 'gain', 0.75)
    np.testing.assert_allclose(result, source * 0.75)
    assert limiter == 1


def test_gain_limits_peaks_without_clipping():
    result, limiter = transform(np.array([0.9, -0.9], dtype=np.float32), 16000, 'gain', 1.5)
    assert np.max(np.abs(result)) <= 0.99001
    assert limiter < 1


@pytest.mark.parametrize('method,value,duration,hz', [
    ('rate', 0.75, 2 / 0.75, 165), ('pitch', -4, 2, 220 * 2 ** (-4/12)),
])
def test_frequency_and_duration(tmp_path, method, value, duration, hz):
    source, sr = sf.read(tone(tmp_path / 'tone.wav'), dtype='float32')
    result, _ = transform(source, sr, method, value)
    assert len(result) / sr == pytest.approx(duration, abs=1/sr)
    assert frequency(result, sr) == pytest.approx(hz, abs=2)


def test_export_backups_manifest_dedup_and_no_enrollment(tmp_path):
    source = tone(tmp_path / 'voice.wav')
    original = source.read_bytes()
    output = export_reference_batch([source, source], tmp_path / 'out', ['rotation', 'gain'])
    assert len(output.outputs) == 2
    assert source.read_bytes() == original
    with ZipFile(output.archive) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 3
        manifest = json.loads(archive.read('manifest.json'))
    assert manifest['automatically_enrolled_or_trained'] is False
    assert manifest['originals_modified'] is False
    for item in manifest['items']:
        assert item['group_id'] == hashlib.sha256(original).hexdigest()
        assert item['identity_preserved'] == 'not_verified'
        assert (output.archive.parent / item['source_backup']).read_bytes() == original
    with pytest.raises(ValueError, match='原始音檔'):
        export_reference_batch(output.outputs, tmp_path / 'out')


@pytest.mark.parametrize('kwargs', [
    {'methods': []}, {'methods': ['bad']}, {'rotation': float('nan')},
    {'rotation': 0}, {'rotation': True}, {'methods': ['gain'], 'gain': 1},
    {'methods': ['pitch'], 'pitch': 0}, {'methods': ['rate'], 'rate': 3},
])
def test_invalid_parameters_create_nothing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        export_reference_batch([], tmp_path / 'out', **kwargs)
    assert not (tmp_path / 'out').exists()


def test_silent_missing_and_empty_inputs(tmp_path):
    sf.write(tmp_path / 'silent.wav', np.zeros(16000), 16000)
    for paths in ([], [tmp_path/'missing.wav'], [tmp_path/'silent.wav']):
        with pytest.raises(ValueError):
            export_reference_batch(paths, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_failed_transform_removes_only_own_batch(monkeypatch, tmp_path):
    from speech_emotion import reference_augmentation as module
    source = tone(tmp_path/'voice.wav')
    root = tmp_path/'out'
    root.mkdir()
    sentinel = root/'keep.txt'
    sentinel.write_text('keep')
    def fail(*args):
        raise ValueError('failed')
    monkeypatch.setattr(module, 'transform', fail)
    with pytest.raises(ValueError):
        export_reference_batch([source], root)
    assert list(root.iterdir()) == [sentinel]
    assert source.is_file()


def test_ui_exports_to_permitted_temp_and_clears_stale_results(monkeypatch, tmp_path):
    import audio_augmentation_tool as tool
    monkeypatch.setattr(tool, 'OUTPUT_ROOT', tmp_path/'outputs')
    result = tool.run_export([str(tone(tmp_path/'voice.wav'))], None,
                             ['rotation'], 40, 0.75, 0.75, -4)
    assert result[0].endswith('.zip')
    assert result[2][:4] == b'RIFF' and result[5][:4] == b'RIFF'
    assert len(result[4]) == 1
    assert '未加入資料庫' in result[-1]
    invalid = tool.run_export([], None, ['rotation'], 40, 0.75, 0.75, -4)
    assert invalid[0] is None and invalid[4] == {} and invalid[5] is None


def test_ui_has_independent_export_and_rotation_default():
    import audio_augmentation_tool as tool
    config = tool.build_app().get_config_file()
    assert any(c['type'] == 'checkboxgroup' and c['props']['value'] == ['rotation']
               for c in config['components'])
    assert any(d.get('api_name') == 'export_audio_experiments' for d in config['dependencies'])
