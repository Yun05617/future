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
    assert "播放回覆" in rendered
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
    assert [components[item]["type"] for item in playback["outputs"]] == ["audio", "markdown"]
    assert playback["js"] is None
    assert components[playback["outputs"][0]]["props"]["autoplay"] is True


def test_companion_portrait_has_transparent_asset_and_responsive_layout():
    from html.parser import HTMLParser
    from urllib.parse import unquote
    from speech_emotion.showcase_ui import (
        COMPANION_HEADER_CSS, COMPANION_PORTRAIT, _companion_header, build_showcase_demo,
    )

    class ImageParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "img":
                self.image = dict(attrs)

    parser = ImageParser()
    parser.feed(_companion_header())
    assert COMPANION_PORTRAIT.is_file()
    from PIL import Image

    with Image.open(COMPANION_PORTRAIT) as portrait:
        assert portrait.mode == "RGBA"
        assert portrait.getchannel("A").getextrema() == (0, 255)
        assert portrait.getpixel((0, 0))[3] == 0
    assert unquote(parser.image["src"]) == "/gradio_api/file=" + COMPANION_PORTRAIT.as_posix()
    assert parser.image["alt"] == "原創韓系女性對話夥伴肖像"
    assert "aspect-ratio:4/5" in COMPANION_HEADER_CSS
    assert "width:176px" in COMPANION_HEADER_CSS
    assert "width:112px" in COMPANION_HEADER_CSS
    assert "border:0; border-radius:0; background:transparent; box-shadow:none" in COMPANION_HEADER_CSS
    assert "mask-image:linear-gradient" in COMPANION_HEADER_CSS
    config = build_showcase_demo().get_config_file()
    header = next(item for item in config["components"] if item["props"].get("elem_id") == "companion-header")
    assert header["props"]["css_template"] == COMPANION_HEADER_CSS
    assert 'data-avatar-state="hidden"' in header["props"]["value"]
    assert 'aria-hidden="true"' in header["props"]["value"]
    assert "opacity:0" in COMPANION_HEADER_CSS
    assert "transition:opacity 2s ease" in COMPANION_HEADER_CSS
    assert "companion-turn-start" in header["props"]["js_on_load"]
    turn = next(item for item in config["dependencies"] if item["api_name"] == "_run_turn")
    assert "Boolean(microphone || upload)" in turn["js"]
    assert any("companion-turn-finish" in (item["js"] or "") for item in config["dependencies"])
    assert "frame.classList.contains('is-visible')) return" in header["props"]["js_on_load"]
    assert "classList.remove" not in header["props"]["js_on_load"]
    assert "companion-reset" not in str(config)


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
    assert loaded[4] == "聽得到嗎"
    assert "聽得到嗎" in loaded[8]["choices"][0][0]
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

    assert len(result) == 11
    assert result[0] == history
    assert "沒有權限" in result[1]
    assert "7862" in result[1]
    assert result[6] == "conversation-1"
    assert result[8]["__type__"] == "update"
    assert result[9:] == ("recording.webm", None)
