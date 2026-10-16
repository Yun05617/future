"""Local, speaker-scoped conversation persistence for the showcase app."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4


ARCHIVE_VERSION = 1
_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class Conversation:
    """One locally persisted chat conversation."""

    id: str
    speaker: str
    created_at: str
    updated_at: str
    title: str = "新對話"
    summary: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    voice_turns: list[dict[str, str | None]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    return text.strip().replace("\n", " ")[:limit]


def _terms(text: str) -> set[str]:
    """Return useful units for mixed Traditional Chinese and Latin search."""
    units: set[str] = set()
    for token in _WORD_RE.findall(text.lower()):
        chinese = [character for character in token if "\u4e00" <= character <= "\u9fff"]
        latin = "".join(character for character in token if character.isascii() and character.isalnum())
        units.update(chinese)
        if latin:
            units.add(latin)
    return units


class ConversationStore:
    """Read and write local conversations below ``專題研究/聊天紀錄``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.directory = self.root / "專題研究" / "聊天紀錄"
        self.archive_path = self.directory / "conversations.json"

    def _read(self) -> list[Conversation]:
        try:
            raw = json.loads(self.archive_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != ARCHIVE_VERSION:
                return []
            records = raw.get("conversations")
            if not isinstance(records, list):
                return []
            return [
                Conversation(**record)
                for record in records
                if isinstance(record, dict)
                and {"id", "speaker", "created_at", "updated_at"}.issubset(record)
            ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    def _write(self, conversations: list[Conversation]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.archive_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"version": ARCHIVE_VERSION, "conversations": [asdict(item) for item in conversations]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.archive_path)

    def list_conversations(self, speaker: str | None = None) -> list[Conversation]:
        conversations = self._read()
        if speaker:
            conversations = [item for item in conversations if item.speaker == speaker]
        return sorted(conversations, key=lambda item: item.updated_at, reverse=True)

    def load_conversation(self, conversation_id: str) -> Conversation | None:
        return next((item for item in self._read() if item.id == conversation_id), None)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Permanently remove one locally stored conversation."""
        if not conversation_id:
            return False
        conversations = self._read()
        remaining = [item for item in conversations if item.id != conversation_id]
        if len(remaining) == len(conversations):
            return False
        self._write(remaining)
        return True

    def create_conversation(self, speaker: str) -> Conversation:
        timestamp = _now()
        conversation = Conversation(uuid4().hex, speaker, timestamp, timestamp)
        conversations = self._read()
        conversations.append(conversation)
        self._write(conversations)
        return conversation

    def save_turn(
        self,
        conversation_id: str,
        speaker: str,
        user_message: str,
        assistant_message: str,
        transcript: str | None,
        emotion: str | None,
        record_id: str | None,
    ) -> Conversation:
        conversations = self._read()
        conversation = next((item for item in conversations if item.id == conversation_id), None)
        if conversation is None:
            timestamp = _now()
            conversation = Conversation(conversation_id, speaker, timestamp, timestamp)
            conversations.append(conversation)
        conversation.speaker = speaker
        conversation.updated_at = _now()
        conversation.messages.extend(
            [{"role": "user", "content": user_message}, {"role": "assistant", "content": assistant_message}]
        )
        if not conversation.title or conversation.title == "新對話":
            conversation.title = _truncate(user_message, 24) or "新對話"
        conversation.summary = _truncate(user_message, 80)
        if transcript or emotion or record_id:
            conversation.voice_turns.append(
                {"record_id": record_id, "transcript": transcript, "emotion": emotion}
            )
        self._write(conversations)
        return conversation

    @staticmethod
    def _searchable(conversation: Conversation) -> str:
        transcript = " ".join(str(turn.get("transcript") or "") for turn in conversation.voice_turns)
        messages = " ".join(item.get("content", "") for item in conversation.messages)
        return " ".join([conversation.title, conversation.summary, transcript, messages])

    def search_conversations(self, query: str, speaker: str | None = None) -> list[Conversation]:
        candidates = self.list_conversations(speaker)
        terms = _terms(query)
        if not terms:
            return candidates
        return [item for item in candidates if all(term in self._searchable(item).lower() for term in terms)]

    def relevant_memories(self, query: str, speaker: str, exclude_id: str | None = None) -> list[str]:
        terms = _terms(query)
        ranked: list[tuple[int, Conversation]] = []
        for conversation in self.list_conversations(speaker):
            if conversation.id == exclude_id:
                continue
            content = self._searchable(conversation)
            score = sum(term in _terms(content) for term in terms)
            if score:
                ranked.append((score, conversation))
        ranked.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [f"{item.speaker}過往對話：{item.title}。{item.summary}" for _, item in ranked[:3]]
