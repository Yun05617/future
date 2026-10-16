"""Restricted client for a local Ollama chat server."""

from urllib.parse import urlparse

import httpx

from speech_emotion.integrated_demo import CombinedAnalysis
from speech_emotion.traditional_chinese import to_traditional_chinese


class OllamaUnavailableError(ConnectionError):
    """Raised when local Ollama is unavailable or an unsafe URL is configured."""


def _chinese_emotion(label: object) -> str:
    return {"positive": "正面", "negative": "負面", "calm": "平靜"}.get(str(label), "未提供")


def build_system_prompt(analysis: CombinedAnalysis, memory_context: list[str] | None = None) -> str:
    """Use the application's voice identity while keeping emotional responses grounded."""
    recognized = bool(analysis.identity and not analysis.identity.is_unknown)
    identity = analysis.identity.speaker if recognized else "未知使用者"
    identity_guidance = (
        "已辨識到成員。首次全名問候由程式另行加上，你的回覆本體不要重複問候或稱呼全名。"
        "後續一般聊天自然用『你』稱呼，不要每次回覆都叫名字，也不要自行猜暱稱。"
        "只有被問『我是誰』『能認出我的聲音嗎』或明確詢問姓名時，先直接回答辨識出的姓名，"
        f"例如『有，我認出你的聲音了，{identity}。』，不要先分析使用者為何提問。"
        if recognized else
        "這次沒有辨識到成員，坦白說『這次還沒辨識出你的聲音，可以再說一段讓我聽聽嗎？』，不可猜名字。"
    )
    emotion = _chinese_emotion((analysis.emotion or {}).get("emotion"))
    transcript = analysis.transcript.effective_text if analysis.transcript else "未提供"
    prompt = (
        "你是具備聲紋身分辨識與情緒感知功能的本機端中文對話夥伴。"
        "身分辨識由系統的聲紋比對模組完成，你負責依據本次辨識結果自然對話，不要自行推翻辨識結果。"
        f"{identity_guidance}"
        "不要主動附加『只是模型推測』『不能辨識真實身分』『不像人類』或『我只是 AI』等能力否認或免責說明。"
        "使用者更正稱呼時請尊重更正；若主動詢問技術或準確度，才簡潔說明是本機聲紋比對，並如實說明限制，不能保證百分之百正確。"
        "不可假裝是真人。本次系統辨識結果優先於歷史助理回覆，不要沿用舊回覆中對身分辨識能力的否認。"
        "聲音情緒欄位只代表聲音模型從音高、語速與節奏推測的結果；逐字稿內容反映的是語意，兩者可能不一致。"
        "若兩者不一致，請明確說『從你說的內容感覺』或『聲音模型目前判斷』，不要把文字語意推測冒充成聲音模型結果。"
        "請用自然、溫暖、有同理心的繁體中文對話。先理解與陪伴使用者的感受，再自然追問一個能幫助了解情況的問題；"
        "除非使用者主動詢問，否則不要一開始就給解決方案、說教或列出步驟。"
        "回覆要像真誠的朋友在傾聽：可以承認事情很受傷或委屈，但不要武斷替使用者下結論。"
        "當使用者談失戀或背叛，先關心發生什麼、感受如何；知道細節後，可以肯定對方受傷不是使用者的錯，"
        "並詢問此刻想被陪伴、想整理想法，還是想討論下一步。"
        "如果使用者明確提到自傷、傷害他人或立即危險，請先鼓勵他聯絡身邊可信任的人與當地緊急協助。\n"
        f"本次系統聲紋身分辨識結果：{identity}\n辨識狀態：{'已辨識成員' if recognized else '尚未辨識成員'}"
        f"\n聲音情緒分析結果：{emotion}\n已校正優先的逐字稿：{transcript}"
    )
    if memory_context:
        prompt += "\n僅供延續脈絡的本機歷史記憶（可能不完整，請勿自行補充事實）：\n" + "\n".join(memory_context)
    return prompt


def _recent_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Keep only valid recent user/assistant turns for a local model request."""
    if not history:
        return []
    valid_messages = []
    # A short recent window keeps the local prompt small enough for quick replies.
    for item in history[-6:]:
        role, content = item.get("role"), item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            valid_messages.append({"role": role, "content": to_traditional_chinese(content.strip())})
    return valid_messages


class OllamaClient:
    """Send structured chat requests only to a loopback Ollama endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise OllamaUnavailableError("Ollama 只能使用本機服務")
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.is_success
        except httpx.HTTPError:
            return False

    def chat(
        self,
        analysis: CombinedAnalysis,
        user_message: str,
        model: str,
        history: list[dict[str, str]] | None = None,
        memory_context: list[str] | None = None,
    ) -> str:
        if not user_message.strip() or not model.strip():
            raise OllamaUnavailableError("請提供聊天內容與本機模型名稱")
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": "10m",
            "think": False,
            "options": {
                "num_predict": 96,
                "num_ctx": 2048,
                "temperature": 0.7,
                "draft_num_predict": 0,
            },
            "messages": [
                {"role": "system", "content": build_system_prompt(analysis, memory_context)},
                *_recent_history(history),
                {"role": "user", "content": to_traditional_chinese(user_message.strip())},
            ],
        }
        try:
            # A 12B local model can need over 30 seconds on first use while its
            # weights are loaded into unified memory; this remains local-only.
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120.0)
            response.raise_for_status()
            reply = str(response.json()["message"]["content"])
            return to_traditional_chinese(reply) if reply.strip() else reply
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise OllamaUnavailableError("本機 Ollama 目前無法回覆") from error
