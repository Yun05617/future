def test_showcase_page_includes_history_sidebar_and_local_voice_controls():
    from pathlib import Path
    from speech_emotion.showcase_ui import BROWSER_RECORDER_HTML, SHOWCASE_CSS, build_showcase_demo

    demo = build_showcase_demo()
    config = demo.get_config_file()
    rendered = str(config)
    assert "具身分辨識與情緒感知的聊天機器人" in rendered
    assert "新增對話" in rendered
    assert "搜尋歷史對話" in rendered
    assert "篩選使用者" in rendered
    assert "歷史對話" in rendered
    assert "重新播放語音" in rendered
    assert "溫柔女聲" in rendered
    assert "中性聲線" in rendered
    assert "flex-wrap:nowrap" in SHOWCASE_CSS
    assert "flex:0 0 44px" in SHOWCASE_CSS
    assert 'sources=["upload"]' in Path("src/speech_emotion/showcase_ui.py").read_text(encoding="utf-8")
    assert 'browser-recorder-frame' in rendered
    assert 'MediaRecorder' in BROWSER_RECORDER_HTML
    assert 'allow="microphone"' in BROWSER_RECORDER_HTML
    assert "#browser-recorder" in SHOWCASE_CSS
    assert "display:none!important" in SHOWCASE_CSS
    assert "#input-dock #audio-input button::after" not in SHOWCASE_CSS
    playback = next(item for item in config["dependencies"] if item["api_name"] == "_play_reply")
    components = {item["id"]: item for item in config["components"]}
    assert [components[item]["type"] for item in playback["outputs"]] == ["audio", "markdown", "state"]
    assert playback["js"] is None
    assert components[playback["outputs"][0]]["props"]["autoplay"] is True


def test_companion_portrait_has_five_animated_character_sets_and_responsive_layout():
    from html.parser import HTMLParser
    from speech_emotion.showcase_ui import (
        CHARACTER_ASSETS, COMPANION_HEADER_CSS, _companion_header, build_showcase_demo,
    )

    class ImageParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images = []

        def handle_starttag(self, tag, attrs):
            if tag == "img":
                self.images.append(dict(attrs))

    parser = ImageParser()
    parser.feed(_companion_header())
    from PIL import Image

    assert set(CHARACTER_ASSETS) == {"姜昀廷", "紀律廷", "趙科翰", "莊媗丞", "未知使用者"}
    for character in CHARACTER_ASSETS.values():
        assert set(character) >= {"base", "talk", "blink"}
        sizes = set()
        for state in ("base", "talk", "blink"):
            path = character[state]
            assert path.is_file()
            with Image.open(path) as portrait:
                assert portrait.mode == "RGBA"
                sizes.add(portrait.size)
        assert len(sizes) == 1
    assert "character-base" in _companion_header()
    assert "character-talk" in _companion_header()
    assert "character-blink" in _companion_header()
    assert parser.images[0]["alt"] == "目前對話使用者的 2D 動態角色"
    assert parser.images[1]["alt"] == parser.images[2]["alt"] == ""
    assert parser.images[1]["aria-hidden"] == parser.images[2]["aria-hidden"] == "true"
    assert "aspect-ratio:4/5" in COMPANION_HEADER_CSS
    assert "width:220px" in COMPANION_HEADER_CSS
    assert "width:128px" in COMPANION_HEADER_CSS
    assert "border:0; border-radius:0; background:transparent; box-shadow:none" in COMPANION_HEADER_CSS
    assert "mask-image:linear-gradient" in COMPANION_HEADER_CSS
    config = build_showcase_demo().get_config_file()
    header = next(item for item in config["components"] if item["props"].get("elem_id") == "companion-header")
    assert header["props"]["css_template"] == COMPANION_HEADER_CSS
    assert 'data-avatar-state="ready"' in header["props"]["value"]
    assert 'aria-hidden="false"' in header["props"]["value"]
    assert "韓系復古紳士風" not in header["props"]["value"]
    assert "溫暖大哥哥風" not in header["props"]["value"]
    assert "準備回覆" not in header["props"]["value"]
    assert "character-talk" in header["props"]["value"]
    assert "character-blink" in header["props"]["value"]
    assert "companion-turn-start" in header["props"]["js_on_load"]
    assert "companion-speech-ready" in header["props"]["js_on_load"]
    assert "目前使用者" in header["props"]["js_on_load"]
    assert "reply-audio" in header["props"]["js_on_load"]
    assert "deepQueryAll(scope, '#reply-audio audio')" in header["props"]["js_on_load"]
    assert "setInterval(syncModel, 200)" in header["props"]["js_on_load"]
    assert "!audio.paused && !audio.ended" in header["props"]["js_on_load"]
    assert "const anyPlaying" in header["props"]["js_on_load"]
    assert "setCharacter" in header["props"]["js_on_load"]
    assert "setAudioLevel" in header["props"]["js_on_load"]
    turn = next(item for item in config["dependencies"] if item["api_name"] == "_run_turn")
    assert "Boolean(microphone || upload)" in turn["js"]
    assert any("companion-turn-finish" in (item["js"] or "") for item in config["dependencies"])
    assert "frame.classList.contains('is-visible')) return" not in header["props"]["js_on_load"]
    assert "classList.remove" not in header["props"]["js_on_load"]
    assert "companion-reset" not in str(config)
    automatic_speech = [
        item for item in config["dependencies"]
        if item["api_name"] == "_play_reply" and item.get("trigger_after") is not None
    ]
    assert automatic_speech, "回覆完成後應自動產生並播放語音"


def test_loading_old_chat_displays_traditional_without_rewriting_archive(monkeypatch, tmp_path):
    from speech_emotion import showcase_ui
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    conversation = store.create_conversation("姜昀廷")
    store.save_turn(conversation.id, "姜昀廷", "听得到吗", "听到了", "听得到吗", "calm", "r1")
    monkeypatch.setattr(showcase_ui, "_STORE", store)
    loaded = showcase_ui._load_chat(conversation.id, "", "全部使用者")
    assert loaded[0][0]["content"] == "聽得到嗎"
    assert loaded[0][1]["content"] == "聽到了"
    assert loaded[4] == "信心度：—"
    assert loaded[5] == "聽得到嗎"
    assert "聽得到嗎" in loaded[9]["choices"][0][0]
    assert store.load_conversation(conversation.id).messages[0]["content"] == "听得到吗"


def test_showcase_runner_loads_browser_controls_as_a_document_script():
    from pathlib import Path

    runner = Path("scripts/run_showcase_demo.py").read_text(encoding="utf-8")

    assert "js=SHOWCASE_JS" not in runner


def test_storage_permission_failure_keeps_history_and_recording(monkeypatch):
    from speech_emotion import showcase_ui

    def denied(*args, **kwargs):
        raise PermissionError("recording storage access denied")

    monkeypatch.setattr(showcase_ui, "run_showcase_chat_turn", denied)
    history = [{"role": "assistant", "content": "原本的對話"}]
    result = showcase_ui._run_turn(
        "", "recording.webm", None, history, "conversation-1", "", "全部使用者",
    )

    assert len(result) == 12
    assert result[0] == history
    assert "沒有權限" in result[1]
    assert "7862" in result[1]
    assert result[4] == "信心度：—"
    assert result[7] == "conversation-1"
    assert result[9]["__type__"] == "update"
    assert result[10:] == ("recording.webm", None)


def test_confidence_tracks_current_classifier_support_and_resets(monkeypatch):
    from types import SimpleNamespace
    from speech_emotion import showcase_ui
    from speech_emotion.speaker_identity import IdentityResult

    analysis = SimpleNamespace(
        identity=IdentityResult(
            "未知使用者", {"姜昀廷": 0.99}, True, 0.0,
            classifier_scores={"姜昀廷": 0.7356, "趙科翰": 0.2644},
        ),
        transcript=None, record_id="record1",
    )
    turn = SimpleNamespace(
        analysis=analysis, history=[], error=None, user_status="目前使用者：未知使用者",
        emotion_status="聲音情緒（模型判斷）：平靜傾向", conversation_id=None,
    )
    monkeypatch.setattr(showcase_ui, "run_showcase_chat_turn", lambda *args, **kwargs: turn)
    monkeypatch.setattr(showcase_ui, "_history_choices", lambda *args: [])
    assert showcase_ui._run_turn("", "voice.wav", None, [], None, "", "全部使用者")[4] == "信心度：73.56%"
    analysis.identity = None
    assert showcase_ui._run_turn("文字", None, None, [], None, "", "全部使用者")[4] == "信心度：—"
    assert showcase_ui._new_chat("", "全部使用者")[4] == "信心度：—"


def test_loading_chat_restores_support_from_its_record(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from speech_emotion import showcase_ui
    from speech_emotion.chat_memory import ConversationStore

    store = ConversationStore(tmp_path)
    conversation = store.create_conversation("姜昀廷")
    store.save_turn(conversation.id, "姜昀廷", "你好", "你好", "你好", "calm", "voice1")
    records = tmp_path / "05_完整分析紀錄" / "records"
    records.mkdir(parents=True)
    (records / "voice1.json").write_text(json.dumps({
        "identity": {"scores": {"姜昀廷": 0.66}, "classifier_scores": {"姜昀廷": 0.980252, "趙科翰": 0.019748}},
    }), encoding="utf-8")
    monkeypatch.setattr(showcase_ui, "_STORE", store)
    monkeypatch.setattr(showcase_ui, "WORKSPACE", SimpleNamespace(root=tmp_path))
    assert showcase_ui._load_chat(conversation.id, "", "全部使用者")[4] == "信心度：98.03%"
    (records / "voice1.json").unlink()
    assert showcase_ui._load_chat(conversation.id, "", "全部使用者")[4] == "信心度：—"
