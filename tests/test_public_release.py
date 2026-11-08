from pathlib import Path

import pytest

from audit_identity_model import session_bounds


def test_no_personal_test_session_is_assumed():
    assert session_bounds(None,None,None) is None


def test_explicit_timezone_session_is_validated():
    from speech_emotion.speaker_identity import KNOWN_SPEAKERS
    low,high=session_bounds('2026-01-01T12:00:00+08:00','2026-01-01T12:10:00+08:00',KNOWN_SPEAKERS[0])
    assert high-low==600


@pytest.mark.parametrize('start,end,speaker',[
    ('2026-01-01T12:00:00+08:00',None,None),
    ('2026-01-01T12:00:00','2026-01-01T12:10:00','unknown'),
    ('2026-01-01T12:00:00+08:00','2026-01-01T11:00:00+08:00','unknown'),
])
def test_partial_or_invalid_session_rejected(start,end,speaker):
    with pytest.raises(ValueError):
        session_bounds(start,end,speaker)


def test_default_workspace_stays_in_repository(monkeypatch):
    import os
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env.pop('VOICE_PROJECT_WORKSPACE',None)
    env['PYTHONPATH']=str(root/'src')
    env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,'-c',
        'from speech_emotion.workspace import DEFAULT_WORKSPACE_ROOT; print(DEFAULT_WORKSPACE_ROOT)'],
        env=env,cwd=root,check=True,capture_output=True,text=True,encoding='utf-8')
    assert Path(result.stdout.strip())==root/'workspace'
