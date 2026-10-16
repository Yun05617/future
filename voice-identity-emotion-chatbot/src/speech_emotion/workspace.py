"""Local project workspace and immutable recording archive helpers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4


DEFAULT_WORKSPACE_ROOT = Path(
    os.environ.get("VOICE_PROJECT_WORKSPACE", str(Path(__file__).resolve().parents[2] / "workspace"))
).expanduser()

WORKSPACE_DIRECTORIES = (
    "01_原始音檔/自動留存",
    "01_原始音檔/身分辨識/莊媗丞",
    "01_原始音檔/身分辨識/姜昀廷",
    "01_原始音檔/身分辨識/紀律廷",
    "01_原始音檔/身分辨識/趙科翰",
    "01_原始音檔/身分辨識/未知使用者測試",
    "02_情緒辨識/正面",
    "02_情緒辨識/負面",
    "02_情緒辨識/平靜",
    "02_情緒辨識/模型",
    "03_語音身分辨識/聲紋向量",
    "03_語音身分辨識/成員聲紋範本",
    "03_語音身分辨識/未知門檻校準",
    "03_語音身分辨識/模型",
    "04_中文語音轉文字/原始轉寫結果",
    "04_中文語音轉文字/人工校正結果",
    "04_中文語音轉文字/正確率報告",
    "05_完整分析紀錄",
    "06_聊天整合/Ollama設定",
    "06_聊天整合/對話紀錄",
    "07_專題文件",
)

JSON_CATEGORIES = {
    "05_完整分析紀錄",
    "04_中文語音轉文字/原始轉寫結果",
    "04_中文語音轉文字/人工校正結果",
    "04_中文語音轉文字/正確率報告",
    "03_語音身分辨識/未知門檻校準",
    "06_聊天整合/Ollama設定",
    "06_聊天整合/對話紀錄",
}


class WorkspaceError(ValueError):
    """Raised when a workspace operation would be unsafe or invalid."""


@dataclass(frozen=True)
class ArchivedAudio:
    """An immutable local copy of a microphone or uploaded recording."""

    record_id: str
    original_path: Path
    created_at: str
    source_kind: str


class ProjectWorkspace:
    """Manage the approved local `專題研究` hierarchy."""

    def __init__(self, root: Path = DEFAULT_WORKSPACE_ROOT) -> None:
        self.root = root.expanduser().resolve()
        for directory in WORKSPACE_DIRECTORIES:
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def archive_input(self, source: Path, source_kind: str) -> ArchivedAudio:
        """Copy one source file to the immutable original-audio archive."""
        source = source.expanduser()
        if source_kind not in {"upload", "microphone"}:
            raise WorkspaceError("unsupported audio source kind")
        if not source.is_file():
            raise WorkspaceError("source audio file does not exist")

        record_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        extension = source.suffix.lower() or ".audio"
        target = self.root / "01_原始音檔" / "自動留存" / f"{record_id}{extension}"
        shutil.copy2(source, target)
        archived = ArchivedAudio(
            record_id=record_id,
            original_path=target,
            created_at=created_at,
            source_kind=source_kind,
        )
        self.write_json(
            "05_完整分析紀錄",
            {
                "record_id": archived.record_id,
                "original_path": str(archived.original_path),
                "source_kind": archived.source_kind,
                "created_at": archived.created_at,
                "status": "archived",
            },
        )
        return archived

    def write_json(self, category: str, payload: dict[str, object]) -> Path:
        """Write a UUID-named JSON record to an approved workspace category."""
        if category not in JSON_CATEGORIES:
            raise WorkspaceError("unsupported workspace JSON category")
        destination_directory = (self.root / category).resolve()
        try:
            destination_directory.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceError("workspace JSON category escapes project root") from error

        destination = destination_directory / f"{uuid4().hex}.json"
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination
