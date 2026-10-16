"""Check portable defaults without accessing real recordings or model weights."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def workspace_default(environment):
    result = subprocess.run(
        [sys.executable, '-c', 'from speech_emotion.workspace import DEFAULT_WORKSPACE_ROOT; print(DEFAULT_WORKSPACE_ROOT)'],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True,
    )
    return Path(result.stdout.strip())


def test_workspace_defaults_to_ignored_repository_directory():
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src')
    environment.pop('VOICE_PROJECT_WORKSPACE', None)
    assert workspace_default(environment) == ROOT / 'workspace'


def test_explicit_private_workspace_is_respected(tmp_path):
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src')
    environment['VOICE_PROJECT_WORKSPACE'] = str(tmp_path / 'private-voice-data')
    assert workspace_default(environment) == tmp_path / 'private-voice-data'


def test_required_local_asset_and_launch_files_are_present():
    for relative in ('assets/companion-portrait-v3.png', 'setup_windows.ps1', 'start_windows.ps1', 'setup_windows.bat', 'start_windows.bat', 'requirements.txt', '.gitignore'):
        assert (ROOT / relative).is_file()
