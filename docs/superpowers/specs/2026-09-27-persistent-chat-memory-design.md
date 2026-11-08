# Persistent Chat Memory Design

## Goal
Turn the competition showcase page into a local two-column chat workspace that preserves conversations by recognised speaker across browser restarts and uses relevant prior content as context for the local Gemma reply.

## User experience
The left sidebar is fixed on desktop and contains the product name, a **新增對話** button, a speaker filter, a search field, and selectable conversation summaries. The main workspace shows the recognised speaker and emotion, current conversation messages, and a fixed composer with text, microphone, and file upload.

Creating a new chat retains the current conversation. Selecting a history item restores its messages. Searches match conversation title, summary, transcript text, and message content. The active speaker filter lists only that speaker's conversations. A voice turn updates the active conversation's speaker to the recognised name; typed-only turns remain `未知使用者` unless the active conversation already has a known speaker.

## Local data and privacy
All chat data is stored locally under `專題研究/聊天紀錄/conversations.json` beneath the existing project workspace. No conversation content is sent to a remote service. The JSON file stores a version number and a list of conversations with: ID, speaker, created/updated timestamps, title, summary, messages, and per-turn voice analysis metadata (record ID, transcript, emotion) where available.

A new conversation title uses the first user message, truncated to 24 characters. Its summary is the most recent user message, truncated to 80 characters. Corrupt or absent JSON must produce an empty history rather than crash the page.

## Memory behavior
Before calling Ollama, the coordinator selects up to three same-speaker past conversations that do not equal the active conversation. It ranks records by simple token overlap between the current user message and each conversation's title, summary, and messages. It passes only the matching text snippets to the existing Ollama client as a dedicated memory-context input. Unknown users never retrieve four members' memories; their history stays under `未知使用者`.

## Technical design
`chat_memory.py` owns JSON serialization, validation, CRUD operations, sidebar query/filtering, and relevance selection. It accepts an explicit base path for deterministic tests. `showcase_chat.py` coordinates a single turn, updates the active conversation in the store, and sends selected memory context to a backward-compatible Ollama client interface. `showcase_ui.py` renders the new layout and binds sidebar controls to the store.

## Error handling
If persistence fails, the current interaction remains visible and the UI reports that the local record could not be saved. Failed audio analysis and unavailable Ollama keep their existing user-facing errors and do not create a partial assistant response.

## Validation
Unit tests cover persistence after store recreation, speaker isolation, search/filtering, corrupt JSON recovery, active-conversation updates, and memory selection. Existing speech, identity, transcription, and showcase tests must remain green. The UI test verifies that the page includes the sidebar controls and main chat controls.
