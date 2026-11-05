# Photo Test Platform Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task.

**Goal:** 可上傳照片與短音訊的本機 EchoMimic 測試頁。
**Architecture:** Gradio UI 呼叫獨立 job service，以 subprocess 啟動現有推論。每次唯一目錄，單工作鎖及取消程序群組，保留錯誤日誌。
**Tech Stack:** Python 3.10, Gradio 3.50.2, Pillow, FFmpeg, existing MPS inference.
**Spec:** docs/superpowers/specs/2026-11-03-photo-test-design.md

## Global Constraints
- 本機資料、不公共分享、不上傳素材至 GitHub。
- 音訊 0.5–5 秒；不承諾一秒出片。

## Review Focus
- 無輸入或壞檔案：在載入模型之前拒絕。
- 長音訊：明確拒絕，不自動裁切。
- 連按生成：單工作鎖。
- 取消：終止本工作程序群組，保留其他程式。
- 舊輸出：唯一工作目錄，只取本次輸出。

### Task 1: Job runner
- [ ] tests/test_photo_platform.py：音訊界限、唯一目錄、互斥與取消測試先失敗。
- [ ] mac/photo_service.py：驗證、隔離、進度、輸出核對及錯誤處理。
- [ ] infer_audio2vid.py：可指定輸出目錄。
- [ ] 跑 unittest 與 py_compile。

### Task 2: UI and launch
- [ ] mac/photo_webui.py：輸入、計時、預覽、取消、下載。
- [ ] mac/start_test_ui.sh 與啟動照片測試平台.command。
- [ ] 本機啟動、HTTP 與瀏覽器驗證；不把舊樣本說成本輪新生成。

### Task 3: Delivery
- [ ] 使用說明、外接硬碟備份、GitHub PR。
- [ ] 獨立檢視差異，修正影響使用的問題。
