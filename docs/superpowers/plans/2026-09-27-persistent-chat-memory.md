# Persistent Chat Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a local, speaker-scoped conversation archive, memory retrieval, and desktop two-column chat workspace.

**Architecture:** A focused `chat_memory.py` JSON store owns persistence and retrieval. The showcase coordinator reads/writes the store around each completed turn, and the Gradio UI renders store-backed sidebar controls while retaining the existing voice and Ollama paths.

**Tech Stack:** Python 3.12, Gradio, JSON, pytest, local Ollama/Gemma.

**Spec:** `docs/superpowers/specs/2026-09-27-persistent-chat-memory-design.md`

## Global Constraints

- Store all content locally at `專題研究/聊天紀錄/conversations.json` beneath the existing workspace.
- Keep speaker memories isolated, including `未知使用者`.
- Do not add remote services, model dependencies, or China-origin components.
- Preserve current local audio, transcription, identity, emotion, and Ollama error behavior.

## Review Focus

- Invalid or corrupted archive JSON must load as an empty archive without losing the ability to start a new chat.
- A known speaker must never receive another known speaker's saved memory snippets.
- Search must include message content and transcript metadata, not only title text.
- A text-only follow-up in a named active conversation must remain associated with that speaker.
- Saving failure must leave the current visible chat intact and report a local persistence error.

---

### Task 1: Local conversation archive

**Files:**
- Create: `src/speech_emotion/chat_memory.py`
- Create: `tests/test_chat_memory.py`

**Interfaces:**
- Produces `ConversationStore(root: Path)`, `create_conversation`, `save_turn`, `list_conversations`, `load_conversation`, `search_conversations`, and `relevant_memories`.

- [ ] **Step 1: Write failing archive tests** for persistence across fresh store instances, corrupt JSON recovery, same-speaker filtering, transcript/message search, and relevant-memory speaker isolation.
- [ ] **Step 2: Run the test module** with `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_memory.py -q`; confirm it fails because the module does not exist.
- [ ] **Step 3: Implement the minimal JSON-backed `ConversationStore`** in `src/speech_emotion/chat_memory.py`, with atomic replacement writes and explicit `Path` input.
- [ ] **Step 4: Re-run the module and full suite**; confirm all pass.
- [ ] **Step 5: Commit** with `feat: add local conversation archive`.

### Task 2: Coordinator persistence and retrieved context

**Files:**
- Modify: `src/speech_emotion/showcase_chat.py`
- Modify: `src/speech_emotion/ollama_client.py`
- Modify: `tests/test_showcase_chat.py`

**Interfaces:**
- Consumes `ConversationStore` from Task 1.
- Extends `run_showcase_chat_turn(..., conversation_id: str | None = None, store: ConversationStore | None = None)` with archive details in `ShowcaseChatResult`.
- Extends the local client protocol with optional `memory_context: list[str] | None`.

- [ ] **Step 1: Write failing coordinator tests** proving a known voice archives speaker and transcript metadata, a text follow-up keeps active speaker, and same-speaker memory is passed to the client.
- [ ] **Step 2: Run targeted tests** and confirm the expected interface failure.
- [ ] **Step 3: Implement the smallest coordinator/client changes** that create/update an active conversation after a successful reply and include no more than three relevant same-speaker snippets in the prompt.
- [ ] **Step 4: Run targeted and full tests**; confirm all pass.
- [ ] **Step 5: Commit** with `feat: persist speaker-scoped chat turns`.

### Task 3: Two-column showcase workspace

**Files:**
- Modify: `src/speech_emotion/showcase_ui.py`
- Modify: `tests/test_showcase_ui.py`

**Interfaces:**
- Consumes the Task 2 archive ID and `ConversationStore` query/load APIs.
- Produces a desktop sidebar with create, search, speaker filter, and conversation selection; the existing main chat controls remain available.

- [ ] **Step 1: Write failing UI tests** checking the sidebar controls and history-loading callbacks are present in the built demo.
- [ ] **Step 2: Run the UI test** and confirm it fails for missing controls.
- [ ] **Step 3: Implement the full-page two-column Gradio layout** with a fixed sidebar and a wide main workspace; make new chat preserve existing history and selected history restore messages/status.
- [ ] **Step 4: Run UI and full tests**; confirm all pass.
- [ ] **Step 5: Commit** with `feat: add searchable chat history sidebar`.

### Task 4: End-to-end verification

**Files:**
- Modify: `README.md` only if launch instructions need adjustment.

- [ ] **Step 1: Run the complete test suite** with `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest -q`.
- [ ] **Step 2: Start the showcase locally** on port 7862 and confirm `curl -fsS http://127.0.0.1:7862/` returns HTTP 200.
- [ ] **Step 3: Commit any concise launch documentation** with `docs: explain local conversation history`.
