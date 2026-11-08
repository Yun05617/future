import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf

from speech_emotion.pitch_augmentation import PitchAugmentationError, export_pitch_batch


def tone(path, frequency=220):
    rate = 16000
    samples = 0.15 * np.sin(2 * np.pi * frequency * np.arange(rate * 2) / rate)
    sf.write(path, samples, rate)
    return path


def dominant_frequency(samples, rate):
    window = np.hanning(len(samples))
    index = np.argmax(np.abs(np.fft.rfft(samples * window)))
    return index * rate / len(samples)


@pytest.mark.parametrize('steps', [4, -4])
def test_real_shift_changes_pitch_and_preserves_original_duration(tmp_path, steps):
    source = tone(tmp_path / 'voice.wav')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = export_pitch_batch([source], tmp_path / 'experimental', steps, 'calm')
    samples, rate = sf.read(result.audio_paths[0])
    assert rate == 16000 and len(samples) == 32000
    assert dominant_frequency(samples, rate) == pytest.approx(220 * 2 ** (steps / 12), abs=3)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    manifest = json.loads(result.manifest_path.read_text(encoding='utf-8'))
    assert manifest['items'][0]['group_id'] == before
    backup = result.manifest_path.parent / manifest['items'][0]['source_backup']
    assert backup.read_bytes() == source.read_bytes()
    assert manifest['items'][0]['identity_preserved'] == 'not_verified'
    assert manifest['items'][0]['emotion_preserved'] == 'not_verified'
    assert manifest['automatically_enrolled_or_trained'] is False
    with ZipFile(result.archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'manifest.json', result.audio_paths[0].name}


@pytest.mark.parametrize('steps', [0, 13, -13, 1.5, float('nan'), float('inf'), True, None])
def test_invalid_steps_are_rejected_before_writing(tmp_path, steps):
    with pytest.raises(PitchAugmentationError):
        export_pitch_batch([], tmp_path / 'experimental', steps)
    assert not (tmp_path / 'experimental').exists()


def test_duplicate_sources_generate_only_one_copy(tmp_path):
    source = tone(tmp_path / 'voice.wav')
    duplicate = tmp_path / 'copy.wav'
    duplicate.write_bytes(source.read_bytes())
    result = export_pitch_batch([source, duplicate, source], tmp_path / 'experimental')
    assert len(result.audio_paths) == 1


def test_missing_or_silent_sources_do_not_create_exports(tmp_path):
    for source in [tmp_path / 'missing.wav', tmp_path / 'silent.wav']:
        if source.name == 'silent.wav':
            sf.write(source, np.zeros(16000), 16000)
        with pytest.raises(PitchAugmentationError):
            export_pitch_batch([source], tmp_path / 'experimental')
    assert not (tmp_path / 'experimental').exists()


def test_export_cannot_feed_its_own_output_back_in(tmp_path):
    source = tone(tmp_path / 'voice.wav')
    result = export_pitch_batch([source], tmp_path / 'experimental')
    with pytest.raises(PitchAugmentationError, match='反覆變調'):
        export_pitch_batch(list(result.audio_paths), tmp_path / 'experimental')


def test_backend_exports_without_enrolling_or_training(monkeypatch, tmp_path):
    from speech_emotion import personal_demo_ui as ui
    from types import SimpleNamespace

    source = tone(tmp_path / 'voice.wav')
    archive = tmp_path / 'batch.zip'
    archive.write_bytes(b'PK-test-archive')
    captured = {}
    monkeypatch.setattr(ui, 'WORKSPACE', SimpleNamespace(root=tmp_path / 'workspace'))
    monkeypatch.setattr(ui.tempfile, 'mkdtemp', lambda prefix: str(tmp_path))
    def export(sources, output_root, **kwargs):
        captured.update(sources=sources, output_root=output_root, **kwargs)
        return SimpleNamespace(archive_path=tmp_path / 'batch.zip', audio_paths=(source,), semitones=4)
    monkeypatch.setattr(ui, 'export_pitch_batch', export)
    path, status = ui._export_pitch_recordings('positive', None, [str(source)], 4)
    assert path.endswith('pitch-augmentation.zip')
    assert Path(path).read_bytes() == archive.read_bytes()
    assert captured['emotion_label'] == 'positive'
    assert captured['semitones'] == 4
    assert '未加入身分範本或情緒訓練' in status


def test_backend_exposes_export_action_with_four_semitones_default():
    from speech_emotion.personal_demo_ui import build_demo
    config = build_demo().get_config_file()
    assert any(c.get('props', {}).get('elem_id') == 'pitch-augmentation' for c in config['components'])
    assert any(c['type'] == 'slider' and c['props'].get('label') == '升降半音' and c['props'].get('value') == 4 for c in config['components'])
    assert any(d.get('api_name') == 'export_pitch_augmentation' for d in config['dependencies'])
