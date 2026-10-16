"""Full-page local chat workspace for the competition showcase."""

from html import escape
import logging
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from speech_emotion.chat_memory import Conversation, ConversationStore
from speech_emotion.personal_demo_ui import (
    KNOWN_SPEAKERS,
    LABEL_TEXT,
    PERSONAL_LABELS,
    SUPPORTED_AUDIO_UPLOAD_TYPES,
    WORKSPACE,
    _SERVICES,
    _save_combined_correction,
)
from speech_emotion.ollama_client import OllamaClient
from speech_emotion.reply_speech import ReplySpeechError, synthesize_reply
from speech_emotion.showcase_chat import run_showcase_chat_turn, traditional_chat_history
from speech_emotion.speaker_identity import UNKNOWN_SPEAKER
from speech_emotion.traditional_chinese import to_traditional_chinese


_ALL_SPEAKERS = "全部使用者"
logger = logging.getLogger(__name__)
_STORE = ConversationStore(WORKSPACE.root.parent)
COMPANION_PORTRAIT = Path(__file__).resolve().parents[2] / "assets" / "companion-portrait-v3.png"

COMPANION_HEADER_CSS = """
.companion-heading { display:grid; grid-template-columns:minmax(0,1fr) 176px; align-items:center; gap:24px; padding:8px 0 14px; }
.companion-copy { min-width:0; color:var(--warm-text,#302b2a); }
.companion-copy .eyebrow { display:block; margin-bottom:10px; color:var(--warm-primary,#bf6857); font-size:11px; font-weight:800; letter-spacing:0; }
.companion-copy h1 { margin:0; font-size:26px; line-height:1.3; font-weight:700; letter-spacing:0; overflow-wrap:anywhere; }
.companion-copy p { margin:10px 0 0; max-width:400px; color:var(--warm-muted,#7c716d); font-size:15px; line-height:1.7; letter-spacing:0; overflow-wrap:anywhere; }
.companion-frame { box-sizing:border-box; width:176px; margin:0; padding:0; border:0; border-radius:0; background:transparent; box-shadow:none; opacity:0; transform:translateY(6px); pointer-events:none; }
.companion-frame.is-visible { opacity:1; transform:translateY(0); transition:opacity 2s ease,transform 2s ease; }
.companion-frame img { display:block; width:100%; height:auto; aspect-ratio:4/5; object-fit:contain; border-radius:0; mask-image:linear-gradient(to bottom,#000 0%,#000 78%,transparent 100%); }
@media (prefers-reduced-motion:reduce) { .companion-frame,.companion-frame.is-visible { transform:none; transition:none; } }
@media (max-width:480px) { .companion-heading { grid-template-columns:minmax(0,1fr) 112px; gap:14px; } .companion-frame { width:112px; } .companion-copy h1 { font-size:21px; } .companion-copy p { font-size:13px; margin-top:8px; } .companion-copy .eyebrow { font-size:10px; line-height:1.5; } }
"""

COMPANION_HEADER_JS = """
const frame = element.querySelector('#companion-avatar');
const controller = new AbortController();
const options = {signal:controller.signal};
window.addEventListener('companion-turn-start', event => {
    if (!event.detail.voice || frame.classList.contains('is-visible')) return;
    frame.dataset.avatarState = 'thinking';
    frame.setAttribute('aria-hidden', 'false');
    frame.classList.add('is-visible');
    frame.scrollIntoView({behavior:'smooth', block:'center'});
}, options);
window.addEventListener('companion-turn-finish', event => {
    if (frame.classList.contains('is-visible')) frame.dataset.avatarState = 'ready';
}, options);
const observer = new MutationObserver(() => {
    if (!element.isConnected) { controller.abort(); observer.disconnect(); }
});
observer.observe(document.body, {childList:true, subtree:true});
"""

COMPANION_START_JS = """(message, microphone, upload, history, conversationId, query, speakerFilter) => {
    window.dispatchEvent(new CustomEvent('companion-turn-start', {detail:{voice:Boolean(microphone || upload)}}));
    return [message, microphone, upload, history, conversationId, query, speakerFilter];
}"""
COMPANION_FINISH_JS = """(status) => {
    window.dispatchEvent(new CustomEvent('companion-turn-finish', {detail:{failed:Boolean(status)}}));
    return [];
}"""


def _companion_header() -> str:
    """Serve only the generated portrait, never the recording workspace."""
    if not COMPANION_PORTRAIT.is_file():
        raise FileNotFoundError(f"Missing companion portrait: {COMPANION_PORTRAIT}")
    gr.set_static_paths(paths=[COMPANION_PORTRAIT])
    image_url = "/gradio_api/file=" + quote(COMPANION_PORTRAIT.as_posix(), safe="/:")
    return (
        '<section class="companion-heading" aria-label="對話夥伴">'
        '<div class="companion-copy"><span class="eyebrow">VOICE · IDENTITY · EMOTION</span>'
        '<h1>開始一段對話</h1><p>我在這裡，陪你把想說的話慢慢說完。</p></div>'
        '<figure class="companion-frame" id="companion-avatar" data-avatar-state="hidden" aria-hidden="true">'
        f'<img src="{escape(image_url, quote=True)}" alt="原創韓系女性對話夥伴肖像" '
        'width="1122" height="1402" loading="eager" decoding="async"></figure></section>'
    )

SHOWCASE_CSS = """
:root { color-scheme:light dark; --warm-bg:#f6f1eb; --warm-panel:#fffdfa; --warm-side:#eee6de; --warm-text:#302b2a; --warm-muted:#7c716d; --warm-primary:#bf6857; --warm-line:rgba(92,70,62,.15); --warm-shadow:0 18px 55px rgba(84,60,51,.10); }
html,body,#root,gradio-app,.gradio-container,.gradio-container > .main { background:var(--warm-bg)!important; color:var(--warm-text)!important; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif!important; }
.gradio-container { width:100%!important; max-width:none!important; min-height:100vh!important; margin:0!important; padding:0!important; }.gradio-container > .main { width:100%!important; max-width:none!important; margin:0!important; padding:0!important; }
#workspace { display:flex!important; flex-wrap:nowrap!important; width:100%!important; min-height:100vh!important; gap:0!important; }
#history-sidebar { flex:0 0 272px!important; min-height:100vh!important; padding:26px 18px!important; background:var(--warm-side)!important; border-right:1px solid var(--warm-line)!important; }
#history-sidebar h2 { margin:0 0 6px!important; font-size:1.25rem!important; letter-spacing:-.03em; } #history-sidebar p { margin:0 0 18px!important; color:var(--warm-muted)!important; font-size:.85rem!important; }
#brand-card { display:flex!important; align-items:center!important; gap:12px!important; margin-bottom:20px!important; padding:14px 15px!important; border:1px solid var(--warm-line)!important; border-radius:18px!important; background:color-mix(in srgb, var(--warm-panel) 76%, transparent)!important; box-shadow:var(--warm-shadow)!important; }
#brand-card .brand-icon { display:grid; place-items:center; flex:0 0 42px; width:42px; height:42px; border-radius:13px; background:var(--warm-primary); color:#fff; font-size:1.25rem; font-weight:700; }
#brand-card h2 { margin:0!important; font-size:1.02rem!important; line-height:1.35!important; letter-spacing:-.025em; } #brand-card p { margin:3px 0 0!important; font-size:.76rem!important; line-height:1.35!important; }
#history-sidebar button { width:100%!important; } #history-sidebar .wrap { margin-top:10px!important; }
#history-list { margin-top:12px!important; } #history-list label { font-size:.82rem!important; color:var(--warm-muted)!important; }
#delete-history { margin-top:8px!important; color:var(--warm-primary)!important; border:1px solid color-mix(in srgb, var(--warm-primary) 38%, transparent)!important; background:transparent!important; }
#chat-main { flex:1 1 0!important; width:auto!important; min-width:0!important; min-height:100vh!important; padding:30px clamp(24px,5vw,88px) 120px!important; background:var(--warm-bg)!important; }
#chat-header { display:flex!important; flex-direction:column!important; width:100%!important; gap:12px!important; margin:0 auto 14px!important; max-width:1120px!important; }
#companion-header { width:100%!important; min-width:0!important; padding:0!important; border:0!important; background:transparent!important; }
.conversation-heading { min-width:0; padding:15px 18px; border:1px solid var(--warm-line); border-radius:18px; background:color-mix(in srgb, var(--warm-panel) 70%, transparent); box-shadow:0 8px 24px rgba(84,60,51,.05); }
.conversation-heading .eyebrow { display:block; margin-bottom:5px; color:var(--warm-primary); font-size:.68rem; font-weight:800; letter-spacing:.14em; }
.conversation-heading h1 { white-space:nowrap; }
.status-chip { display:block; padding:0; border:0; border-radius:0; background:transparent; color:var(--warm-muted); font-size:.86rem; white-space:nowrap; }
#chat-status { width:100%!important; gap:8px 20px!important; padding:10px 0!important; border-top:1px solid var(--warm-line)!important; border-bottom:1px solid var(--warm-line)!important; }
#chat-header #chat-status .message-panel { flex:1 1 240px!important; min-width:0!important; min-height:0!important; padding:0!important; border:0!important; background:transparent!important; box-shadow:none!important; }
#chat-status .status-chip { white-space:normal!important; overflow-wrap:anywhere; }
#chat-header > .message-panel .prose, #chat-header > .message-panel .markdown { margin:0!important; }
#showcase-chat { width:100%!important; max-width:1120px!important; height:clamp(340px,62vh,720px)!important; min-height:300px!important; margin:0 auto!important; border:1px solid var(--warm-line)!important; border-radius:22px!important; background:color-mix(in srgb, var(--warm-panel) 82%, transparent)!important; box-shadow:var(--warm-shadow)!important; } #showcase-chat .wrap { border:0!important; background:transparent!important; } #showcase-chat .message { border-radius:18px!important; } #showcase-chat .message.user { background:rgba(191,104,87,.14)!important; }
#input-dock { position:sticky!important; bottom:14px!important; z-index:5!important; align-items:center!important; width:100%!important; max-width:1120px!important; margin:14px auto 0!important; padding:8px!important; border:1px solid var(--warm-line)!important; border-radius:20px!important; background:color-mix(in srgb, var(--warm-panel) 94%, transparent)!important; box-shadow:var(--warm-shadow)!important; backdrop-filter:blur(18px); } #input-dock > .form { min-width:0!important; } #input-dock .block { min-width:0!important; } #input-dock textarea { min-height:42px!important; height:42px!important; }
#input-dock #audio-input { display:none!important; } #input-dock #upload-input { flex:0 0 44px!important; width:44px!important; min-width:44px!important; max-width:44px!important; height:44px!important; overflow:hidden!important; border:0!important; } #audio-input label,#upload-input label { display:none!important; }
#browser-recorder { flex:0 0 48px!important; width:48px!important; min-width:48px!important; height:48px!important; } #browser-recorder iframe { display:block!important; width:48px!important; height:48px!important; border:0!important; overflow:hidden!important; }
#input-dock #upload-input .upload-container { width:44px!important; min-width:44px!important; min-height:44px!important; height:44px!important; padding:0!important; border:0!important; border-radius:50%!important; background:transparent!important; } #input-dock #upload-input [data-testid="upload-text"] { font-size:0!important; } #input-dock #upload-input [data-testid="upload-text"]::before { content:"＋"; color:var(--warm-text); font-size:28px; font-weight:300; } #input-dock #upload-input [data-testid="upload-icon"] { display:none!important; } #input-dock #upload-input .file-preview { max-height:44px!important; overflow:hidden!important; }
#voice-recorder-status { display:block!important; min-height:18px!important; margin:5px 4px 0!important; color:var(--warm-muted)!important; font-size:.78rem!important; line-height:1.35!important; }
#input-dock button.primary { flex:0 0 44px!important; width:44px!important; min-width:44px!important; height:44px!important; min-height:44px!important; padding:0!important; border-radius:50%!important; font-size:0!important; } #input-dock button.primary::after { content:"↑"; font-size:24px; line-height:1; }
.utility-row { margin:8px 0!important; align-items:center!important; } .utility-row button { min-height:34px!important; font-size:.85rem!important; }.message-panel { min-height:0!important; padding:0!important; border:0!important; background:transparent!important; }
.gradio-container button { border-radius:12px!important; min-height:40px; font-weight:650!important; }.gradio-container button.primary { background:var(--warm-primary)!important; border:0!important; }
/* Gradio's queue badge is useful while debugging, but competes with the chat in the public demo. */
.progress-text, .queue-status, [data-testid="queue-status"], .eta-bar { display:none!important; }
@media (max-width: 760px) { #workspace { flex-direction:column!important; flex-wrap:nowrap!important; } #history-sidebar { flex:auto!important; min-height:auto!important; border-right:0!important; border-bottom:1px solid var(--warm-line)!important; } #chat-main { min-height:auto!important; padding:18px 14px 90px!important; } #chat-header { flex-direction:column!important; } .conversation-heading h1 { white-space:normal!important; } #showcase-chat { height:52vh!important; } }
@media (prefers-color-scheme: dark) { :root { --warm-bg:#211e1d; --warm-panel:#2c2827; --warm-side:#292421; --warm-text:#fbf5f1; --warm-muted:#cbbdb5; --warm-line:rgba(255,236,227,.15); } #input-dock { background:color-mix(in srgb, var(--warm-panel) 94%, transparent)!important; } }
"""

BROWSER_RECORDER_DOCUMENT = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><style>
html,body{margin:0;width:48px;height:48px;background:transparent;overflow:hidden}button{width:48px;height:48px;border:0;border-radius:50%;background:transparent;color:#302b2a;font-size:20px;cursor:pointer;transition:background .16s,color .16s}button.recording{background:#bf6857;color:#fff}button:focus-visible{outline:2px solid #7b3e32;outline-offset:2px}
</style></head><body><button id="record" type="button" aria-label="開始錄音" title="開始錄音">🎙</button><script>
let stream=null, recorder=null, chunks=[];
const button=document.getElementById('record');
function setButton(recording){button.classList.toggle('recording',recording);button.textContent=recording?'■':'🎙';button.setAttribute('aria-label',recording?'停止錄音':'開始錄音');button.title=recording?'停止錄音':'開始錄音'}
function setStatus(text){const root=window.parent.document;const host=root.querySelector('gradio-app');const scope=host&&host.shadowRoot?host.shadowRoot:root;const status=scope.querySelector('#voice-input-status');if(status)status.textContent=text}
function deliver(blob){const root=window.parent.document;const host=root.querySelector('gradio-app');const scope=host&&host.shadowRoot?host.shadowRoot:root;const input=scope.querySelector('#audio-input input[type=file]');if(!input){setStatus('無法接收錄音，請重新整理頁面。');return}const file=new File([blob],'voice-message.webm',{type:'audio/webm'});const data=new DataTransfer();data.items.add(file);input.files=data.files;input.dispatchEvent(new Event('change',{bubbles:true,composed:true}));setStatus('錄音已完成，按送出即可分析。')}
async function toggle(){if(recorder&&recorder.state==='recording'){setStatus('正在完成錄音…');recorder.stop();return}try{setStatus('正在要求麥克風權限…');stream=await navigator.mediaDevices.getUserMedia({audio:true});chunks=[];recorder=new MediaRecorder(stream);recorder.ondataavailable=e=>chunks.push(e.data);recorder.onstop=()=>{const blob=new Blob(chunks,{type:'audio/webm'});if(blob.size)deliver(blob);else setStatus('沒有收到聲音，請再試一次。');stream.getTracks().forEach(track=>track.stop());setButton(false)};recorder.start();setButton(true);setStatus('正在錄音，再按一次即可停止。')}catch(error){setButton(false);setStatus(error&&error.name==='NotAllowedError'?'請允許瀏覽器使用麥克風後再試一次。':'無法啟動麥克風。')}}
button.addEventListener('click',toggle);
</script></body></html>"""

BROWSER_RECORDER_HTML = (
    '<iframe id="browser-recorder-frame" title="語音輸入" allow="microphone" '
    f'srcdoc="{escape(BROWSER_RECORDER_DOCUMENT, quote=True)}"></iframe>'
)


def _history_choices(query: str = "", speaker: str = _ALL_SPEAKERS) -> list[tuple[str, str]]:
    selected_speaker = None if speaker == _ALL_SPEAKERS else speaker
    conversations = _STORE.search_conversations(query or "", selected_speaker)
    return [(to_traditional_chinese(f"{item.speaker} · {item.title or '新對話'}"), item.id) for item in conversations]


def _history_update(query: str, speaker: str):
    return gr.update(choices=_history_choices(query, speaker), value=None)


def _conversation_status(conversation: Conversation) -> tuple[str, str, str]:
    latest_turn = conversation.voice_turns[-1] if conversation.voice_turns else {}
    emotion = latest_turn.get("emotion") if latest_turn else None
    emotion_text = {"positive": "正面傾向", "negative": "負面傾向", "calm": "平靜傾向"}.get(emotion, "未分析")
    transcript = to_traditional_chinese(str(latest_turn.get("transcript") or "尚未分析語音。"))
    return f"目前使用者：{conversation.speaker}", f"聲音情緒（模型判斷）：{emotion_text}", transcript


def _new_chat(query: str, speaker_filter: str):
    return (
        [], "", "目前使用者：未知使用者", "聲音情緒（模型判斷）：未分析", "尚未分析語音。", "", None,
        UNKNOWN_SPEAKER, gr.update(choices=_history_choices(query, speaker_filter), value=None),
    )


def _load_chat(conversation_id: str | None, query: str, speaker_filter: str):
    conversation = _STORE.load_conversation(conversation_id or "")
    if conversation is None:
        return _new_chat(query, speaker_filter)
    user_status, emotion_status, transcript = _conversation_status(conversation)
    return (
        traditional_chat_history(conversation.messages), "", user_status, emotion_status, transcript, "", conversation.id,
        conversation.speaker, gr.update(choices=_history_choices(query, speaker_filter), value=conversation.id),
    )


def _delete_chat(conversation_id: str | None, query: str, speaker_filter: str):
    """Delete the selected history item and reset the chat view."""
    deleted = _STORE.delete_conversation(conversation_id or "")
    message = "已刪除選取的歷史對話。" if deleted else "請先選取要刪除的歷史對話。"
    return (
        [], message, "目前使用者：未知使用者", "聲音情緒（模型判斷）：未分析", "尚未分析語音。", "", None,
        UNKNOWN_SPEAKER, gr.update(choices=_history_choices(query, speaker_filter), value=None),
    )


def _play_reply(history: list[dict[str, str]], voice_mode: str):
    text = next((item["content"] for item in reversed(history or []) if item.get("role") == "assistant"), "")
    try:
        audio_path = synthesize_reply(text, voice_mode, WORKSPACE.root / "06_聊天整合" / "回覆語音")
    except ReplySpeechError as error:
        return None, str(error)
    return str(audio_path), ""


def _run_turn(message, microphone, upload, history, conversation_id, query, speaker_filter):
    try:
        result = run_showcase_chat_turn(
            message, microphone, upload, history or [], WORKSPACE, _SERVICES, OllamaClient(),
            conversation_id=conversation_id, store=_STORE,
        )
    except PermissionError:
        logger.exception("Chat service cannot access recording or conversation storage")
        return (
            history or [],
            "目前服務沒有權限存取錄音或對話資料。請從檔案總管雙擊「啟動全部_Windows.bat」，"
            "看到 Ready 後使用 http://127.0.0.1:7862/。本次錄音尚未完成分析。",
            "目前使用者：未知使用者", "聲音情緒（模型判斷）：未分析",
            "尚未分析語音。", "", conversation_id, UNKNOWN_SPEAKER,
            gr.update(), microphone, upload,
        )
    analysis = result.analysis
    transcript = analysis.transcript.effective_text if analysis and analysis.transcript else "尚未分析語音。"
    record_id = analysis.record_id if analysis else ""
    speaker = result.user_status.removeprefix("目前使用者：")
    return (
        result.history, result.error or "", result.user_status, result.emotion_status, transcript, record_id,
        result.conversation_id, speaker, gr.update(choices=_history_choices(query, speaker_filter), value=result.conversation_id),
        None, None,
    )


def build_showcase_demo() -> gr.Blocks:
    """Build the public-facing local competition chat workspace."""
    with gr.Blocks(title="具身分辨識與情緒感知的聊天機器人") as demo:
        with gr.Row(elem_id="workspace"):
            with gr.Column(elem_id="history-sidebar", scale=2):
                gr.HTML("<div id='brand-card'><div class='brand-icon' aria-hidden='true'>✦</div><div><h2>具身分辨識與情緒感知的聊天機器人</h2><p>本機端語音身分與情緒感知</p></div></div>")
                new_chat = gr.Button("＋ 新增對話", variant="primary")
                history_search = gr.Textbox(label="搜尋歷史對話", placeholder="搜尋內容、逐字稿或標題")
                speaker_filter = gr.Dropdown(
                    choices=[_ALL_SPEAKERS, *KNOWN_SPEAKERS, UNKNOWN_SPEAKER], value=_ALL_SPEAKERS,
                    label="篩選使用者",
                )
                history_picker = gr.Radio(choices=_history_choices(), label="歷史對話", elem_id="history-list")
                delete_chat = gr.Button("刪除選取對話", variant="secondary", elem_id="delete-history")
                gr.Markdown("[模型管理與訓練後台](http://127.0.0.1:7860/)")
            with gr.Column(elem_id="chat-main", scale=7):
                with gr.Column(elem_id="chat-header"):
                    gr.HTML(
                        _companion_header(), elem_id="companion-header",
                        css_template=COMPANION_HEADER_CSS, apply_default_css=False,
                        js_on_load=COMPANION_HEADER_JS,
                    )
                    with gr.Row(elem_id="chat-status"):
                        user_status = gr.Markdown("<span class='status-chip'>目前使用者：未知使用者</span>", elem_classes=["message-panel"])
                        emotion_status = gr.Markdown("<span class='status-chip'>聲音情緒（模型判斷）：未分析</span>", elem_classes=["message-panel"])
                history_view = gr.Chatbot(show_label=False, elem_id="showcase-chat", placeholder="你好，我在這裡陪你聊聊。")
                history_state = gr.State([])
                conversation_state = gr.State(None)
                record_state = gr.State("")
                with gr.Row(elem_id="input-dock", equal_height=True):
                    message = gr.Textbox(show_label=False, placeholder="輸入訊息，或使用語音和我聊聊…", lines=1, scale=8)
                    microphone = gr.Audio(sources=["upload"], type="filepath", show_label=False, elem_id="audio-input", scale=1)
                    upload = gr.File(file_types=SUPPORTED_AUDIO_UPLOAD_TYPES, type="filepath", show_label=False, elem_id="upload-input", scale=1)
                    gr.HTML(BROWSER_RECORDER_HTML, elem_id="browser-recorder", scale=1)
                    send = gr.Button("送出", variant="primary", scale=1, elem_id="send-message")
                gr.HTML("<span id='voice-input-status' role='status'>點擊麥克風開始錄音。</span>")
                status = gr.Markdown("", elem_classes=["message-panel"])
                with gr.Row(elem_classes=["utility-row"]):
                    voice_mode = gr.Radio(choices=[("溫柔女聲", "gentle"), ("中性聲線", "neutral")], value="gentle", label="回覆語音")
                    play = gr.Button("播放回覆")
                    clear = gr.Button("清除目前內容")
                reply_audio = gr.Audio(
                    label="回覆語音", type="filepath", format="wav",
                    interactive=False, autoplay=True, elem_id="reply-audio",
                )
                with gr.Accordion("查看這次語音分析與更正", open=False):
                    transcript = gr.Textbox(label="本次繁體中文轉寫", interactive=False, lines=3)
                    speaker = gr.Dropdown(choices=[*KNOWN_SPEAKERS, UNKNOWN_SPEAKER], value=UNKNOWN_SPEAKER, label="正確說話者")
                    emotion = gr.Dropdown(choices=[(LABEL_TEXT[key], key) for key in PERSONAL_LABELS], value="calm", label="正確情緒")
                    correction = gr.Textbox(label="正確逐字稿（可留白）", lines=2)
                    save = gr.Button("儲存更正並更新模型")
                    correction_status = gr.Markdown("")

        refresh_outputs = [history_view, status, user_status, emotion_status, transcript, record_state, conversation_state, speaker, history_picker]
        history_search.change(_history_update, inputs=[history_search, speaker_filter], outputs=history_picker)
        speaker_filter.change(_history_update, inputs=[history_search, speaker_filter], outputs=history_picker)
        new_chat.click(_new_chat, inputs=[history_search, speaker_filter], outputs=refresh_outputs).then(lambda: [], outputs=history_state).then(lambda: (None, None), outputs=[microphone, upload])
        history_picker.change(_load_chat, inputs=[history_picker, history_search, speaker_filter], outputs=refresh_outputs).then(lambda h: h, inputs=history_view, outputs=history_state)
        delete_chat.click(_delete_chat, inputs=[conversation_state, history_search, speaker_filter], outputs=refresh_outputs).then(lambda: [], outputs=history_state).then(lambda: (None, None), outputs=[microphone, upload])
        send.click(
            _run_turn, inputs=[message, microphone, upload, history_state, conversation_state, history_search, speaker_filter],
            outputs=[*refresh_outputs, microphone, upload], js=COMPANION_START_JS,
        ).then(lambda h: h, inputs=history_view, outputs=history_state).then(
            fn=None, inputs=status, outputs=[], js=COMPANION_FINISH_JS, queue=False,
        )
        clear.click(lambda: ([], "", "目前使用者：未知使用者", "聲音情緒（模型判斷）：未分析", "尚未分析語音。", "", None, UNKNOWN_SPEAKER, gr.update(value=None)), outputs=refresh_outputs).then(lambda: [], outputs=history_state).then(lambda: (None, None), outputs=[microphone, upload])
        play.click(_play_reply, inputs=[history_state, voice_mode], outputs=[reply_audio, status])
        for action in (new_chat, delete_chat, clear, send):
            action.click(lambda: None, outputs=reply_audio, queue=False)
        history_picker.change(lambda: None, outputs=reply_audio, queue=False)
        save.click(_save_combined_correction, inputs=[record_state, speaker, emotion, correction], outputs=[correction_status, user_status, emotion_status, transcript])
    return demo
