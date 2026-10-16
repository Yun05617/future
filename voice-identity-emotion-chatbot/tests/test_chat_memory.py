from speech_emotion.speaker_identity import UNKNOWN_SPEAKER


def test_conversation_persists_after_store_is_recreated(tmp_path):
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    conversation = store.create_conversation("莊媗丞")
    store.save_turn(
        conversation.id,
        speaker="莊媗丞",
        user_message="我失戀了",
        assistant_message="我在這裡聽你說。",
        transcript="我失戀了",
        emotion="negative",
        record_id="record-1",
    )

    reloaded = ConversationStore(tmp_path).load_conversation(conversation.id)

    assert reloaded is not None
    assert reloaded.speaker == "莊媗丞"
    assert reloaded.messages[-1]["content"] == "我在這裡聽你說。"
    assert reloaded.title == "我失戀了"


def test_corrupt_archive_recovers_as_empty_history(tmp_path):
    from speech_emotion.chat_memory import ConversationStore

    archive = tmp_path / "專題研究" / "聊天紀錄" / "conversations.json"
    archive.parent.mkdir(parents=True)
    archive.write_text("not json", encoding="utf-8")

    assert ConversationStore(tmp_path).list_conversations() == []


def test_search_matches_transcript_and_keeps_speakers_isolated(tmp_path):
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    zhuang = store.create_conversation("莊媗丞")
    store.save_turn(zhuang.id, "莊媗丞", "今天報告壓力很大", "我陪你整理。", "報告壓力很大", "negative", "r1")
    jiang = store.create_conversation("姜昀廷")
    store.save_turn(jiang.id, "姜昀廷", "我也擔心報告", "我們一起想。", "報告", "calm", "r2")

    found = store.search_conversations("壓力", speaker="莊媗丞")
    memories = store.relevant_memories("報告讓我好焦慮", speaker="莊媗丞", exclude_id="new")

    assert [item.id for item in found] == [zhuang.id]
    assert len(memories) == 1
    assert "莊媗丞" in memories[0]
    assert "姜昀廷" not in memories[0]
    assert store.search_conversations("報告", speaker=UNKNOWN_SPEAKER) == []


def test_delete_conversation_removes_selected_history(tmp_path):
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    conversation = store.create_conversation("莊媗丞")

    assert store.delete_conversation(conversation.id) is True
    assert store.load_conversation(conversation.id) is None
    assert store.delete_conversation(conversation.id) is False
