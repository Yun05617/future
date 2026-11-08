from pathlib import Path

import pytest

from speech_emotion.integrated_demo import CombinedAnalysis
from speech_emotion.ollama_client import OllamaClient, OllamaUnavailableError, build_system_prompt
from speech_emotion.speaker_identity import IdentityResult
from speech_emotion.transcription import TranscriptResult


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "我在這裡聽你說。"}}


@pytest.fixture
def analysis():
    return CombinedAnalysis(
        "record-1", Path("archive.wav"), Path("processed.wav"),
        IdentityResult("莊媗丞", {"莊媗丞": 0.9}, False, 1.0),
        {"emotion": "calm"}, TranscriptResult("record-1", "原始文字", "校正文字", 0.1, 1.0),
        2.0, "ok",
    )


def test_chat_sends_identity_emotion_and_corrected_transcript(monkeypatch, analysis):
    captured = {}
    monkeypatch.setattr(
        "speech_emotion.ollama_client.httpx.post",
        lambda url, json, timeout: captured.update({"url": url, "json": json}) or FakeResponse(),
    )
    reply = OllamaClient().chat(analysis, "我今天心情不好", "local-chat-model")

    assert "莊媗丞" in captured["json"]["messages"][0]["content"]
    assert "平靜" in captured["json"]["messages"][0]["content"]
    assert "校正文字" in captured["json"]["messages"][0]["content"]
    assert reply == "我在這裡聽你說。"


def test_chat_includes_recent_conversation_and_empathic_guidance(monkeypatch, analysis):
    captured = {}
    monkeypatch.setattr(
        "speech_emotion.ollama_client.httpx.post",
        lambda url, json, timeout: captured.update({"json": json}) or FakeResponse(),
    )

    OllamaClient().chat(
        analysis,
        "他出軌了",
        "local-chat-model",
        history=[
            {"role": "user", "content": "我失戀了"},
            {"role": "assistant", "content": "你們怎麼了？願意說說看嗎？"},
        ],
    )

    messages = captured["json"]["messages"]
    system_prompt = messages[0]["content"]
    assert "先理解與陪伴" in system_prompt
    assert "不要一開始就給解決方案" in system_prompt
    assert messages[1:] == [
        {"role": "user", "content": "我失戀了"},
        {"role": "assistant", "content": "你們怎麼了？願意說說看嗎？"},
        {"role": "user", "content": "他出軌了"},
    ]


def test_chat_rejects_non_local_service(analysis):
    with pytest.raises(OllamaUnavailableError):
        OllamaClient(base_url="https://example.com").chat(analysis, "測試", "local-chat-model")


def test_prompt_uses_recognized_identity_without_default_capability_disclaimer(analysis):
    prompt = build_system_prompt(analysis)
    assert "本次系統聲紋身分辨識結果：莊媗丞" in prompt
    assert "辨識狀態：已辨識成員" in prompt
    assert "先直接回答辨識出的姓名" in prompt
    assert "不要主動附加" in prompt
    assert "才簡潔說明是本機聲紋比對" in prompt
    assert "不可把它們當作事實" not in prompt
    assert "目前推測身分" not in prompt
    assert "首次全名問候由程式另行加上" in prompt
    assert "後續一般聊天自然用『你』稱呼" in prompt


def test_model_output_and_input_are_converted_to_traditional(monkeypatch, analysis):
    captured = {}

    class SimplifiedResponse(FakeResponse):
        def json(self):
            return {"message": {"content": "听得到你的声音。"}}

    monkeypatch.setattr(
        "speech_emotion.ollama_client.httpx.post",
        lambda url, json, timeout: captured.update({"json": json}) or SimplifiedResponse(),
    )
    reply = OllamaClient().chat(analysis, "听得到吗", "local", history=[{"role": "user", "content": "开心"}])
    assert reply == "聽得到你的聲音。"
    assert captured["json"]["messages"][-1]["content"] == "聽得到嗎"
    assert captured["json"]["messages"][1]["content"] == "開心"


def test_prompt_does_not_treat_rejected_identity_as_recognized(analysis):
    from dataclasses import replace

    rejected = replace(analysis, identity=IdentityResult("莊媗丞", {"莊媗丞": 0.3}, True, 1.0))
    prompt = build_system_prompt(rejected)
    assert "本次系統聲紋身分辨識結果：未知使用者" in prompt
    assert "辨識狀態：尚未辨識成員" in prompt
    assert "不可猜名字" in prompt
