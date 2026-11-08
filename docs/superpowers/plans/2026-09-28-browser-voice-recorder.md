# Browser Voice Recorder Implementation Plan

**Goal:** Add a reliable toggle microphone recorder to the showcase chat input.

**Spec:** `docs/superpowers/specs/2026-09-28-browser-voice-recorder-design.md`

1. Add a failing UI test for custom recorder markup and bundled script asset.
2. Add an external JavaScript asset loaded by a normal script URL, not dynamically injected head content.
3. Implement MediaRecorder start/stop, microphone errors, and transfer of the final WebM File into hidden Gradio audio input.
4. Verify with pytest, a local HTTP check, and browser DOM inspection.
