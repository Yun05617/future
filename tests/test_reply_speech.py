import json
import os
import subprocess
import wave

import pytest

from speech_emotion import reply_speech


@pytest.mark.skipif(os.name != "nt", reason="Windows speech adapter")
def test_reply_text_is_data_and_repeated_playback_uses_cached_audio(monkeypatch, tmp_path):
    calls = []
    text = '你好。 "quoted" $(literal-text)\n第二行'

    def fake_run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        calls.append(payload)
        assert command[0].endswith("powershell.exe")
        assert text not in " ".join(command)
        with wave.open(payload["output_path"], "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x01\x00" * 1600)

    monkeypatch.setattr(reply_speech.subprocess, "run", fake_run)
    output = reply_speech.synthesize_reply(text, "gentle", tmp_path)
    assert reply_speech.synthesize_reply(text, "gentle", tmp_path) == output
    assert len(calls) == 1
    assert calls[0]["text"] == text
    assert output.read_bytes()[:4] == b"RIFF"


@pytest.mark.skipif(os.name != "nt", reason="Windows speech adapter")
def test_failed_synthesis_removes_partial_audio(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr=b"Speech engine unavailable")

    monkeypatch.setattr(reply_speech.subprocess, "run", fail)
    with pytest.raises(reply_speech.ReplySpeechError):
        reply_speech.synthesize_reply("你好", "neutral", tmp_path)
    assert not list(tmp_path.glob("*.wav"))


def test_play_button_returns_inline_message_when_no_reply_exists(monkeypatch):
    from speech_emotion.showcase_ui import _play_reply

    def synthesize(text, mode, directory):
        assert text == ""
        raise reply_speech.ReplySpeechError("尚無可播放的回覆")

    monkeypatch.setattr("speech_emotion.showcase_ui.synthesize_reply", synthesize)
    assert _play_reply([], "gentle") == (None, "尚無可播放的回覆")
