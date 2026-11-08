from pathlib import Path

from speech_emotion.integrated_demo import CombinedAnalysis
from speech_emotion.speaker_identity import IdentityResult, UNKNOWN_SPEAKER
from speech_emotion.transcription import TranscriptResult


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, analysis, user_message, model, history=None, memory_context=None):
        self.calls.append({"analysis": analysis, "message": user_message, "model": model, "history": history, "memory_context": memory_context})
        return "我在這裡聽你說。"


def _analysis(speaker: str) -> CombinedAnalysis:
    return CombinedAnalysis(
        "record-1",
        Path("raw.wav"),
        Path("processed.wav"),
        IdentityResult(speaker, {speaker: 0.9}, speaker == UNKNOWN_SPEAKER, 1.0),
        {"emotion": "negative", "scores": {"negative": 0.8}},
        TranscriptResult("record-1", "我失恋了", None, None, 1.0, "我失戀了"),
        1.0,
        "ok",
    )


def test_first_known_voice_turn_adds_name_greeting(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat

    monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_: _analysis("莊媗丞"))
    result = showcase_chat.run_showcase_chat_turn(
        "", "voice.wav", None, [], tmp_path, object(), FakeClient()
    )

    assert result.user_status == "目前使用者：莊媗丞"
    assert result.greeting_added is True
    assert "莊媗丞你好" in result.history[-1]["content"]
    assert result.history[0]["content"] == "我失戀了"


def test_unknown_voice_never_adds_name_greeting(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat

    monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_: _analysis(UNKNOWN_SPEAKER))
    result = showcase_chat.run_showcase_chat_turn(
        "", "unknown.wav", None, [], tmp_path, object(), FakeClient()
    )

    assert result.user_status == "目前使用者：未知使用者"
    assert result.greeting_added is False
    assert "你好" not in result.history[-1]["content"].split("。", 1)[0]


def test_analysis_failure_keeps_conversation_id_and_returns_inline_error(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat

    failed = CombinedAnalysis(
        "record-failed", Path("raw.webm"), None, None, None, None, 1.0,
        "error", "聲紋模型無法載入",
    )
    monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_: failed)

    result = showcase_chat.run_showcase_chat_turn(
        "", "voice.webm", None, [], tmp_path, object(), FakeClient(),
        conversation_id="conversation-1",
    )

    assert result.conversation_id == "conversation-1"
    assert result.error == "聲紋模型無法載入"


def test_second_turn_sends_existing_history_to_local_model(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat

    monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_: _analysis("莊媗丞"))
    client = FakeClient()
    first = showcase_chat.run_showcase_chat_turn("我失戀了", None, None, [], tmp_path, object(), client)
    showcase_chat.run_showcase_chat_turn("他出軌了", None, None, first.history, tmp_path, object(), client)

    assert client.calls[-1]["history"] == first.history


def test_known_voice_turn_is_saved_with_transcript_metadata(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat
    from speech_emotion.chat_memory import ConversationStore

    monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_: _analysis("莊媗丞"))
    store = ConversationStore(tmp_path)
    result = showcase_chat.run_showcase_chat_turn(
        "", "voice.wav", None, [], tmp_path, object(), FakeClient(), store=store
    )

    saved = store.load_conversation(result.conversation_id)
    assert saved is not None
    assert saved.speaker == "莊媗丞"
    assert saved.voice_turns[-1]["transcript"] == "我失戀了"


def test_active_named_conversation_keeps_speaker_and_passes_memory(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    earlier = store.create_conversation("莊媗丞")
    store.save_turn(earlier.id, "莊媗丞", "報告壓力很大", "我陪你。", "報告壓力很大", "negative", "old")
    active = store.create_conversation("莊媗丞")
    client = FakeClient()
    result = showcase_chat.run_showcase_chat_turn(
        "我還是很擔心報告", None, None, [], tmp_path, object(), client,
        conversation_id=active.id, store=store,
    )

    assert result.user_status == "目前使用者：莊媗丞"
    assert client.calls[-1]["memory_context"] == ["莊媗丞過往對話：報告壓力很大。報告壓力很大"]


def test_first_successful_recognition_after_unknown_turn_announces_name(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    active = store.create_conversation(UNKNOWN_SPEAKER)
    active = store.save_turn(active.id, UNKNOWN_SPEAKER, "聽得到嗎", "聽得到。", "聽得到嗎", "calm", "first")
    monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_: _analysis("姜昀廷"))
    result = showcase_chat.run_showcase_chat_turn(
        "", "voice.wav", None, active.messages, tmp_path, object(), FakeClient(),
        conversation_id=active.id, store=store,
    )
    assert result.greeting_added is True
    assert result.history[-1]["content"].startswith("姜昀廷你好")
    assert store.load_conversation(active.id).speaker == "姜昀廷"


def test_each_member_is_greeted_once_even_after_unknown_or_another_speaker(monkeypatch, tmp_path):
    from speech_emotion import showcase_chat
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    history, conversation_id = [], None
    for speaker, expected in [("姜昀廷", True), ("姜昀廷", False), (UNKNOWN_SPEAKER, False),
                              ("姜昀廷", False), ("紀律廷", True), ("姜昀廷", False)]:
        monkeypatch.setattr(showcase_chat, "analyse_recording", lambda *_, name=speaker: _analysis(name))
        result = showcase_chat.run_showcase_chat_turn(
            "", "voice.wav", None, history, tmp_path, object(), FakeClient(),
            conversation_id=conversation_id, store=store,
        )
        assert result.greeting_added is expected
        history, conversation_id = result.history, result.conversation_id
    assert sum(item["content"].startswith("姜昀廷你好") for item in history) == 1


def test_chat_normalizes_user_reply_and_gradio_history(tmp_path):
    from speech_emotion import showcase_chat
    from speech_emotion.chat_memory import ConversationStore

    class SimplifiedClient(FakeClient):
        def chat(self, *args, **kwargs):
            super().chat(*args, **kwargs)
            return "我听到了，你今天很开心。"

    client = SimplifiedClient()
    store = ConversationStore(tmp_path)
    old_history = [{"role": "assistant", "content": [{"type": "text", "text": "继续聊聊。"}]}]
    result = showcase_chat.run_showcase_chat_turn(
        "今天很开心", None, None, old_history, tmp_path, object(), client, store=store,
    )
    assert result.history == [
        {"role": "assistant", "content": "繼續聊聊。"},
        {"role": "user", "content": "今天很開心"},
        {"role": "assistant", "content": "我聽到了，你今天很開心。"},
    ]
    assert client.calls[-1]["message"] == "今天很開心"
    assert old_history[0]["content"][0]["text"] == "继续聊聊。"
    assert store.load_conversation(result.conversation_id).messages[-1]["content"] == result.history[-1]["content"]
