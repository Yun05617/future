# 聲緒聊天助理競賽主頁 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立溫暖陪伴感的競賽展示聊天室，同時保留現有模型管理與訓練後台。

**Architecture:** 新增獨立的 Gradio 展示應用與啟動腳本，展示頁只負責歡迎、聊天、狀態顯示與可展開分析；既有 `personal_demo_ui.py` 繼續負責訓練與管理。將中文轉寫的原始文字、繁體顯示文字與人工校正文字明確分離，聊天協調邏輯抽出成可測試的單位。

**Tech Stack:** Python 3.12、Gradio 6、OpenAI Whisper `small`、Apple Foundation/PyObjC、本機 Ollama + Google Gemma 4 12B、SpeechBrain、Meta XLS-R、瀏覽器 Web Speech API。

**Spec:** `docs/superpowers/specs/2026-09-27-showcase-chat-assistant-design.md`

## Global Constraints

- 所有推論僅能使用本機端服務與模型；Ollama 只能使用 `127.0.0.1` 或 `localhost`。
- 不得使用中國或中資背景的模型、API、MCP 或介接工具；不得加入 emotion2vec、SenseVoice、Qwen 或 DeepSeek。
- Whisper 使用 OpenAI 官方 `small` 權重；Gemma 使用 Google Gemma 4 12B；情緒模型維持 Meta XLS-R 特徵與團隊自訓分類器。
- 繁體顯示與語音回覆僅使用 macOS／瀏覽器的本機系統能力，不傳送文字或聲音至外部服務。
- 原始音檔、原始轉寫、繁體顯示結果、人工校正與訓練資料繼續保存在桌面「專題研究」。
- 現有 `scripts/run_integrated_demo.py` 與模型管理、訓練、校正流程必須保留可用。

## Review Focus

- 未知使用者首次語音輸入：頂端顯示「未知使用者」，不得產生姓名問候；由 Task 3 測試。
- 已知使用者後續語音輸入：只在新對話的第一則成功辨識時加上姓名問候；由 Task 3 測試。
- macOS 繁體轉換不可用：保留原始 Whisper 文字並回傳可理解的狀態，不使聊天中斷；由 Task 1 測試。
- Whisper 快取尚未下載或模型載入失敗：介面顯示下一步提示而非 Python 例外；由 Task 2 測試。
- 系統找不到指定中文語音：文字回覆與聊天紀錄仍可使用，播放控制顯示不可用原因；由 Task 4 測試。

---

## File Structure

| 檔案 | 變更責任 |
|---|---|
| `src/speech_emotion/traditional_chinese.py` | 封裝 macOS Foundation 的繁體轉換與可理解錯誤。 |
| `src/speech_emotion/transcription.py` | 採用 Whisper `small`，保存原始、繁體與校正轉寫。 |
| `src/speech_emotion/integrated_demo.py` | 將原始轉寫服務接到繁體轉換流程。 |
| `src/speech_emotion/showcase_chat.py` | 可測試的展示聊天回合協調、首次身分問候與狀態資料。 |
| `src/speech_emotion/showcase_ui.py` | 新的歡迎頁、聊天主頁、可展開分析與本機語音回覆 UI。 |
| `src/speech_emotion/personal_demo_ui.py` | 只調整現有後台以顯示繁體優先的轉寫結果。 |
| `scripts/run_showcase_demo.py` | 在獨立展示連接埠啟動競賽主頁。 |
| `tests/test_traditional_chinese.py` | 繁體轉換單元測試。 |
| `tests/test_transcription.py` | 原始／繁體／校正轉寫紀錄測試。 |
| `tests/test_showcase_chat.py` | 身分問候、未知使用者、對話上下文與錯誤處理測試。 |
| `tests/test_showcase_ui.py` | 展示 UI 的文字、輸入元件與本機語音選項測試。 |
| `requirements.txt` | 加入 Apple Foundation 的 Python bridge，並鎖定相容版本。 |

### Task 1: 繁體中文轉換與轉寫資料結構

**Files:**
- Create: `src/speech_emotion/traditional_chinese.py`
- Modify: `src/speech_emotion/transcription.py`
- Modify: `requirements.txt`
- Test: `tests/test_traditional_chinese.py`
- Test: `tests/test_transcription.py`

**Interfaces:**
- Produces: `to_traditional_chinese(text: str) -> str`，以及 `TraditionalChineseError(ValueError)`。
- Produces: `TranscriptResult.traditional_text: str | None` 與 `TranscriptResult.effective_text: str`；優先順序為人工校正、繁體結果、原始結果。
- Consumes: macOS Foundation 的繁簡轉換能力；非 macOS 或轉換失敗時提出 `TraditionalChineseError`。

- [ ] **Step 1: 寫失敗測試，指定繁體轉換與轉寫文字優先順序**

```python
def test_to_traditional_chinese_converts_simplified_text():
    assert to_traditional_chinese("我失恋了") == "我失戀了"

def test_effective_transcript_prefers_correction_then_traditional_text():
    result = TranscriptResult("r1", "我失恋了", None, None, 1.0, "我失戀了")
    assert result.effective_text == "我失戀了"
```

- [ ] **Step 2: 執行失敗測試**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_traditional_chinese.py tests/test_transcription.py -q`  
Expected: FAIL，因 `to_traditional_chinese`、`traditional_text` 或 `effective_text` 尚未存在。

- [ ] **Step 3: 實作 `traditional_chinese.py` 與更新 `TranscriptResult`**

`to_traditional_chinese` 僅接受非空字串，使用 Apple Foundation 的 Simplified-to-Traditional transform；將平台或轉換錯誤包裝為繁體中文錯誤。將 `traditional_text` 加在 dataclass 最後且提供預設值，以維持舊的五欄 JSON 和既有 positional test fixtures 相容。更新 JSON 讀取、原始轉寫寫入與校正儲存，使原始 Whisper 文字不被覆寫。

- [ ] **Step 4: 執行聚焦測試與完整測試**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest -q`  
Expected: PASS；舊 JSON 紀錄仍能載入，繁體輸出優先顯示。

- [ ] **Step 5: 提交**

```bash
git add requirements.txt src/speech_emotion/traditional_chinese.py src/speech_emotion/transcription.py tests/test_traditional_chinese.py tests/test_transcription.py
git commit -m "feat: display Whisper transcripts in Traditional Chinese"
```

### Task 2: Whisper small 與整合分析錯誤訊息

**Files:**
- Modify: `src/speech_emotion/transcription.py`
- Modify: `src/speech_emotion/integrated_demo.py`
- Modify: `src/speech_emotion/personal_demo_ui.py`
- Test: `tests/test_transcription.py`
- Test: `tests/test_integrated_demo.py`

**Interfaces:**
- Consumes: `OpenAIWhisperTranscriber(model_name: str = "small")`。
- Produces: `TranscriptError` 的使用者可理解原因，例如「Whisper 模型尚未下載完成或無法載入」。
- Produces: 整合分析與後台 UI 一律顯示 `TranscriptResult.effective_text`。

- [ ] **Step 1: 寫失敗測試，指定預設模型與模型載入錯誤**

```python
def test_openai_whisper_defaults_to_small_model():
    assert OpenAIWhisperTranscriber().model_name == "small"

def test_transcribe_chinese_reports_model_load_failure(tmp_path, workspace):
    with pytest.raises(TranscriptError, match="Whisper 模型"):
        transcribe_chinese(tmp_path / "voice.wav", "r1", workspace, BrokenTranscriber())
```

- [ ] **Step 2: 執行失敗測試**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transcription.py -q`  
Expected: FAIL，預設仍是 `base` 或錯誤沒有可理解原因。

- [ ] **Step 3: 實作模型設定與錯誤翻譯**

保留 OpenAI Whisper 官方下載與快取機制，但把預設改為 `small`。僅在模型載入／下載例外時回傳具體下一步說明；不得關閉 SSL 驗證或下載非官方權重。整合分析捕捉 `TranscriptError` 並讓頁面顯示繁體錯誤，不顯示 Python stack trace。

- [ ] **Step 4: 執行測試與手動快取驗證**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest -q`  
Expected: PASS。

Run: `PYTHONPATH=src .venv/bin/python -c 'from speech_emotion.transcription import OpenAIWhisperTranscriber; print(OpenAIWhisperTranscriber().model_name)'`  
Expected: `small`。

- [ ] **Step 5: 提交**

```bash
git add src/speech_emotion/transcription.py src/speech_emotion/integrated_demo.py src/speech_emotion/personal_demo_ui.py tests/test_transcription.py tests/test_integrated_demo.py
git commit -m "feat: use Whisper small for local Chinese transcription"
```

### Task 3: 展示聊天室回合協調與姓名問候

**Files:**
- Create: `src/speech_emotion/showcase_chat.py`
- Modify: `src/speech_emotion/ollama_client.py`
- Test: `tests/test_showcase_chat.py`

**Interfaces:**
- Produces: `ShowcaseChatResult(history: list[dict[str, str]], analysis: CombinedAnalysis, user_status: str, emotion_status: str, greeting_added: bool)`。
- Produces: `run_showcase_chat_turn(message: str, microphone_path: str | None, uploaded_path: str | None, history: list[dict[str, str]], workspace: ProjectWorkspace, services: AnalysisServices, client: OllamaClient) -> ShowcaseChatResult`。
- Consumes: 現有 `analyse_recording`、`CombinedAnalysis`、`OllamaClient.chat(..., history=...)`。

- [ ] **Step 1: 寫失敗測試，指定已知／未知身分與兩輪上下文**

```python
def test_first_known_voice_turn_adds_name_greeting(fake_services, fake_client):
    result = run_showcase_chat_turn("", "voice.wav", None, [], workspace, fake_services, fake_client)
    assert result.user_status == "目前使用者：莊媗丞"
    assert result.greeting_added is True
    assert "莊媗丞你好" in result.history[-1]["content"]

def test_unknown_voice_never_adds_name_greeting(fake_services, fake_client):
    result = run_showcase_chat_turn("", "unknown.wav", None, [], workspace, fake_services, fake_client)
    assert result.user_status == "目前使用者：未知使用者"
    assert "你好" not in result.history[-1]["content"].split("。", 1)[0]
```

- [ ] **Step 2: 執行失敗測試**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_showcase_chat.py -q`  
Expected: FAIL，模組與函式尚未存在。

- [ ] **Step 3: 實作 `run_showcase_chat_turn`**

將音訊選取、完整分析、有效逐字稿選擇、最近聊天歷史、Gemma 回覆與顯示狀態集中在此函式。只在 `history` 為空、輸入為成功語音分析且身分不是 `未知使用者` 時，為助理回覆加上姓名問候。文字輸入與未知聲音不加姓名；模型失敗時回傳可顯示的狀態且不破壞既有歷史。

- [ ] **Step 4: 執行聚焦與完整測試**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest -q`  
Expected: PASS；既有多輪 Ollama 歷史測試維持通過。

- [ ] **Step 5: 提交**

```bash
git add src/speech_emotion/showcase_chat.py src/speech_emotion/ollama_client.py tests/test_showcase_chat.py
git commit -m "feat: add showcase chat orchestration and identity greeting"
```

### Task 4: 競賽展示主頁與本機語音回覆

**Files:**
- Create: `src/speech_emotion/showcase_ui.py`
- Create: `scripts/run_showcase_demo.py`
- Test: `tests/test_showcase_ui.py`

**Interfaces:**
- Produces: `build_showcase_demo() -> gr.Blocks`。
- Produces: `SHOWCASE_CSS`，包含暖色淺色／深色模式、歡迎頁、狀態列、聊天泡泡與展開面板。
- Consumes: `run_showcase_chat_turn`、既有 `_save_combined_correction` 或等價的公開更正服務。

- [ ] **Step 1: 寫失敗測試，指定展示頁的必要可見文案與控制項**

```python
def test_showcase_page_includes_welcome_and_local_voice_controls():
    demo = build_showcase_demo()
    config = demo.get_config_file()
    assert "聲緒聊天助理" in str(config)
    assert "開始聊天" in str(config)
    assert "播放回覆" in str(config)
    assert "溫柔女聲" in str(config)
    assert "中性聲線" in str(config)
```

- [ ] **Step 2: 執行失敗測試**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_showcase_ui.py -q`  
Expected: FAIL，展示 UI 尚未存在。

- [ ] **Step 3: 建立歡迎頁與聊天頁**

用同一個 `gr.Blocks` 的可見區塊切換，預設只顯示歡迎頁，按「開始聊天」後顯示聊天室。聊天室包含錄音、上傳、文字輸入、目前使用者、目前情緒、聊天紀錄、清除按鈕與「查看這次語音分析」折疊區；後台入口連至既有 `http://127.0.0.1:7861/`。

- [ ] **Step 4: 加入本機回覆語音控制**

以瀏覽器 Web Speech API 播放助理回覆文字，透過 macOS 可用的中文系統語音播放，不傳送文字至外部服務。提供「溫柔女聲／中性聲線」選擇與「播放回覆」按鈕；找不到可用中文語音時，停用播放並顯示原因，文字回覆不受影響。將 JavaScript 以 Gradio `js` load hook 綁定到展示頁專用元素，避免影響後台。

- [ ] **Step 5: 執行測試與本機手動驗證**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest -q`  
Expected: PASS。

Run: `PYTHONPATH=src .venv/bin/python scripts/run_showcase_demo.py --port 7862`  
Expected: 本機開啟 `http://127.0.0.1:7862/`，先顯示歡迎頁。

Manual: 以已知成員錄音、未知錄音、文字輸入、兩輪失戀情境、兩個語音模式與折疊校正區逐一驗證。

- [ ] **Step 6: 提交**

```bash
git add src/speech_emotion/showcase_ui.py scripts/run_showcase_demo.py tests/test_showcase_ui.py
git commit -m "feat: add warm showcase chat assistant page"
```

### Task 5: 後台繁體顯示與展示／後台共存驗證

**Files:**
- Modify: `src/speech_emotion/personal_demo_ui.py`
- Modify: `scripts/run_integrated_demo.py`（僅在需要更新文案時）
- Test: `tests/test_personal_demo.py`
- Test: `tests/test_showcase_ui.py`

**Interfaces:**
- Consumes: `TranscriptResult.effective_text`、既有更正、訓練、撤回與聲紋函式。
- Produces: 後台顯示繁體優先轉寫，展示頁與後台可同時以不同連接埠啟動。

- [ ] **Step 1: 寫失敗測試，指定後台優先顯示繁體文字**

```python
def test_combined_analysis_format_uses_effective_traditional_transcript():
    formatted = _format_combined_analysis(analysis_with_traditional_text)
    assert "我失戀了" in formatted[3]
    assert "我失恋了" not in formatted[3]
```

- [ ] **Step 2: 執行失敗測試**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_personal_demo.py -q`  
Expected: FAIL，既有格式化函式仍直接使用 `original_text`。

- [ ] **Step 3: 更新後台格式與更正預填值**

所有對使用者顯示、傳入 Ollama 的轉寫文字採 `effective_text`；原始辨識文字仍只在分析檔與資料欄位中保留。確認訓練資料、撤回、更正、聲紋範本與舊後台按鈕未被移除。

- [ ] **Step 4: 執行完整測試與雙服務 smoke test**

Run: `MPLCONFIGDIR=.matplotlib PYTHONPATH=src .venv/bin/python -m pytest -q`  
Expected: PASS。

Run: `PYTHONPATH=src .venv/bin/python scripts/run_integrated_demo.py --port 7861` and `PYTHONPATH=src .venv/bin/python scripts/run_showcase_demo.py --port 7862`  
Expected: `curl -fsS http://127.0.0.1:7861/` and `curl -fsS http://127.0.0.1:7862/` both return HTTP 200.

- [ ] **Step 5: 提交**

```bash
git add src/speech_emotion/personal_demo_ui.py scripts/run_integrated_demo.py tests/test_personal_demo.py tests/test_showcase_ui.py
git commit -m "feat: preserve Traditional Chinese across demo surfaces"
```

## Plan Review

- **Spec coverage:** Tasks 1–2 cover繁體與 Whisper small；Task 3 covers身分問候、情緒線索與多輪對話；Task 4 covers歡迎頁、暖色 UI、輸入、折疊分析、語音回覆與後台入口；Task 5 covers既有後台保留、資料優先順序與雙服務驗證。
- **Type consistency:** 轉寫資料由 `TranscriptResult` 統一；展示聊天由 `ShowcaseChatResult` 統一；UI 僅消費此結果。
- **Review focus coverage:** 五個高風險情況各自歸入 Tasks 1–4 的明確測試。
- **Scope:** 本計畫不更動模型來源、臉部辨識或雲端部署；保持焦點在本機競賽展示頁與可靠的語音中文體驗。
