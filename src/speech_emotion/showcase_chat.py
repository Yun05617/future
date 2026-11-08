"""Conversation orchestration for the competition showcase page."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from speech_emotion.chat_memory import ConversationStore
from speech_emotion.integrated_demo import AnalysisServices, CombinedAnalysis, analyse_recording
from speech_emotion.ollama_client import OllamaUnavailableError
from speech_emotion.speaker_identity import IdentityResult, UNKNOWN_SPEAKER
from speech_emotion.transcription import TranscriptResult
from speech_emotion.traditional_chinese import to_traditional_chinese
from speech_emotion.workspace import ProjectWorkspace


MODEL_NAME = "gemma4:12b"
_EMOTION_TEXT = {"positive": "正面傾向", "negative": "負面傾向", "calm": "平靜傾向"}


def traditional_chat_history(history: list[dict]) -> list[dict[str, str]]:
    """Accept persisted strings and Gradio's text blocks without altering originals."""
    messages = []
    for item in history or []:
        content = item.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text", "")) for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if isinstance(content, str) and item.get("role") in {"user", "assistant"}:
            messages.append({
                "role": item["role"],
                "content": to_traditional_chinese(content) if content.strip() else content,
            })
    return messages


class LocalChatClient(Protocol):
    def chat(
        self,
        analysis: CombinedAnalysis,
        user_message: str,
        model: str,
        history: list[dict[str, str]] | None = None,
        memory_context: list[str] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class ShowcaseChatResult:
    """One render-ready local chat turn."""

    history: list[dict[str, str]]
    analysis: CombinedAnalysis | None
    user_status: str
    emotion_status: str
    greeting_added: bool
    conversation_id: str | None = None
    error: str | None = None


def _text_only_analysis(message: str) -> CombinedAnalysis:
    return CombinedAnalysis(
        record_id="text-only",
        archive_path=Path(),
        processed_path=None,
        identity=IdentityResult(UNKNOWN_SPEAKER, {}, True, 0.0),
        emotion=None,
        transcript=TranscriptResult("text-only", message, None, None, 0.0, message),
        total_latency_ms=0.0,
        status="ok",
    )


def _voice_input(
    microphone_path: str | None, uploaded_path: str | None
) -> tuple[Path | None, str | None]:
    if uploaded_path:
        return Path(uploaded_path), "upload"
    if microphone_path:
        return Path(microphone_path), "microphone"
    return None, None


def _status_for(analysis: CombinedAnalysis) -> tuple[str, str]:
    speaker = analysis.identity.speaker if analysis.identity else UNKNOWN_SPEAKER
    emotion = (analysis.emotion or {}).get("emotion")
    return f"目前使用者：{speaker}", f"聲音情緒（模型判斷）：{_EMOTION_TEXT.get(str(emotion), '未分析')}"


def run_showcase_chat_turn(
    message: str,
    microphone_path: str | None,
    uploaded_path: str | None,
    history: list[dict[str, str]],
    workspace: ProjectWorkspace,
    services: AnalysisServices,
    client: LocalChatClient,
    conversation_id: str | None = None,
    store: ConversationStore | None = None,
) -> ShowcaseChatResult:
    """Analyse one optional recording and generate a context-aware local reply."""
    previous_history = traditional_chat_history(history)
    source, source_kind = _voice_input(microphone_path, uploaded_path)
    typed_message = message.strip()
    if typed_message:
        typed_message = to_traditional_chinese(typed_message)
    if source is not None:
        analysis = analyse_recording(source, source_kind or "microphone", workspace, services)
        if analysis.status != "ok":
            return ShowcaseChatResult(
                history=previous_history,
                analysis=analysis,
                user_status="目前使用者：未知使用者",
                emotion_status="聲音情緒（模型判斷）：未分析",
                greeting_added=False,
                conversation_id=conversation_id,
                error=analysis.message or "語音分析失敗，請再試一次。",
            )
        user_message = typed_message or (analysis.transcript.effective_text if analysis.transcript else "")
    elif typed_message:
        analysis = _text_only_analysis(typed_message)
        user_message = typed_message
    else:
        return ShowcaseChatResult(
            history=previous_history,
            analysis=None,
            user_status="目前使用者：未知使用者",
            emotion_status="聲音情緒（模型判斷）：未分析",
            greeting_added=False,
            conversation_id=conversation_id,
            error="請輸入文字，或錄音／上傳音檔。",
        )
    if not user_message:
        return ShowcaseChatResult(
            history=previous_history,
            analysis=analysis,
            user_status=_status_for(analysis)[0],
            emotion_status=_status_for(analysis)[1],
            greeting_added=False,
            conversation_id=conversation_id,
            error="沒有取得可用的對話內容，請重新錄音或直接輸入文字。",
        )
    speaker = analysis.identity.speaker if analysis.identity else UNKNOWN_SPEAKER
    active_conversation = store.load_conversation(conversation_id) if store and conversation_id else None
    if source is None and active_conversation is not None:
        speaker = active_conversation.speaker
        analysis = CombinedAnalysis(
            analysis.record_id, analysis.archive_path, analysis.processed_path,
            IdentityResult(speaker, {}, speaker == UNKNOWN_SPEAKER, 0.0), analysis.emotion,
            analysis.transcript, analysis.total_latency_ms, analysis.status, analysis.message,
        )
    if store and active_conversation is None:
        active_conversation = store.create_conversation(speaker)
    memory_context = store.relevant_memories(user_message, speaker, active_conversation.id) if store and active_conversation else []
    try:
        reply = client.chat(analysis, user_message, MODEL_NAME, history=previous_history, memory_context=memory_context)
        if reply.strip():
            reply = to_traditional_chinese(reply)
    except OllamaUnavailableError as error:
        return ShowcaseChatResult(
            history=previous_history,
            analysis=analysis,
            user_status=_status_for(analysis)[0],
            emotion_status=_status_for(analysis)[1],
            greeting_added=False,
            conversation_id=active_conversation.id if active_conversation else conversation_id,
            error=f"本機 Gemma 暫時無法回覆：{error}",
        )
    greeting_history = [
        *previous_history,
        *traditional_chat_history(active_conversation.messages if active_conversation else []),
    ]
    already_greeted = any(
        item["role"] == "assistant" and item["content"].startswith(f"{speaker}你好")
        for item in greeting_history
    )
    greeting_added = bool(source and speaker != UNKNOWN_SPEAKER and not already_greeted)
    if greeting_added:
        reply = f"{speaker}你好，{reply}"
    updated_history = [
        *previous_history,
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    if store and active_conversation:
        active_conversation = store.save_turn(
            active_conversation.id, speaker, user_message, reply,
            analysis.transcript.effective_text if analysis.transcript else None,
            str((analysis.emotion or {}).get("emotion")) if analysis.emotion else None,
            analysis.record_id if source else None,
        )
    return ShowcaseChatResult(
        updated_history,
        analysis,
        *_status_for(analysis),
        greeting_added,
        active_conversation.id if active_conversation else conversation_id,
    )
