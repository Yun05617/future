# 具身分辨識與情緒感知的聊天機器人

A Chatbot with Identity Recognition and Emotion Awareness

本機端繁體中文語音聊天專題。整合聲紋身分辨識、三類聲音情緒分析、中文語音轉文字與 Ollama 對話，包含 Gradio 展示前台及訓練後台。

這份版本只收錄程式、測試、必要介面圖片與文件。**不包含已訓練的個人模型、成員聲紋範本、錄音或聊天紀錄**。重新下載後，需要建立自己的訓練資料；不能直接取得原展示電腦的辨識結果。

## Windows 快速開始

1. 安裝 **Python 3.12（64 位元）**、**FFmpeg** 及 **Ollama**。FFmpeg 的 `bin` 目錄需加入 `PATH`。這份相依版本以現有 Python 3.12 環境為準，其他 Python 版本未驗證。
2. 下載或 clone 本專案，解壓縮至可寫入的資料夾。
3. 雙擊 `setup_windows.bat` 安裝 Python 套件。
4. 啟動 Ollama，在 PowerShell 執行：

   ```powershell
   ollama pull gemma4:12b
   ollama list
   ffmpeg -version
   ```

5. 雙擊 `start_windows.bat`，等待終端機顯示服務網址。**保留終端機視窗**；要停止服務時按 `Ctrl+C`。

| 入口 | 本機網址 |
| --- | --- |
| 聊天展示前台 | http://127.0.0.1:7862/ |
| 模型訓練後台 | http://127.0.0.1:7860/ |

網址只在執行程式的電腦上有效，不是 GitHub Pages 網站。第一次使用相關語音模型會下載權重；之後使用本機快取。RTX 4060 可供 Ollama 視條件使用，部分語音模組目前仍以 CPU 執行。

詳細安裝、單獨啟動及資料移轉方式見 [Windows 操作說明](docs/SETUP_WINDOWS.md)。

## 第一次訓練

- 在後台分別加入 `positive`（正面）、`negative`（負面）、`calm`（平靜）錄音，每類至少兩段，再執行情緒分類器訓練。這是流程最低需求，不代表足以得到可靠準確率。
- 登錄各成員的身分錄音並建立／更新聲紋範本。預設成員姓名在 `src/speech_emotion/speaker_identity.py`；工作區資料夾結構在 `workspace.py`，兩處需保持一致。
- 目前採最高餘弦相似度 **大於 0.40** 顯示候選成員，未通過顯示未知使用者。分數不是辨識正確率；此暫行門檻可能誤認，不能當作安全登入驗證。
- 檢查逐字稿、情緒與身分結果後，再測試完整語音對話。

## 功能範圍

- 錄音、音訊上傳、文字輸入及繁體中文轉錄。
- 本機聲紋比對、三類聲音情緒分類與個人對話紀錄。
- Ollama／Google Gemma 對話，首次全名稱呼後自然延續聊天。
- 透明角色圖片首次漸入，之後持續顯示。
- Windows 語音合成介接程式；實際播放取決於系統語音、瀏覽器及權限，仍需實機驗證。
- 歷史四類情緒 baseline 的訓練及預測 CLI 也保留於 `scripts/`，不應與三類展示分類器混用。

這次整理不包含另外的臉部識別程式；角色嘴型同步亦尚未在此原始碼中實作。詳見 [目前狀態](docs/PROJECT_STATUS.md)。

## 專案結構

```text
voice-identity-emotion-chatbot/
  src/speech_emotion/      語音、身分、情緒、對話與介面程式
  scripts/                展示、訓練及檢查入口
  tests/                  Python 測試及 PowerShell 檢查
  assets/                 介面使用的透明角色圖
  data/manifests/         不含真實錄音的範例清單
  reports/provenance.md   既有模型與資料來源紀錄
  docs/                   安裝、狀態與資料管理說明
  requirements.txt        現有環境的套件版本
  pytest.ini              測試路徑設定
  setup_windows.*         建立虛擬環境並安裝相依套件
  start_windows.*         啟動本機服務
```

`workspace/`、`.venv/`、`.cache/`、`data/raw/`、`data/processed/`、`models/` 等執行資料不納入 Git。資料夾會在執行時建立。

## 測試

於專案根目錄執行，不需要先啟動聊天伺服器：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
powershell.exe -NoProfile -File tests\test_windows_server_process.ps1
powershell.exe -NoProfile -File tests\test_repository_scripts.ps1
```

測試多數使用合成音訊與模擬推論，不等於實際聲紋準確率、真實語音播放或大型模型效能驗證。FFmpeg 若未安裝，部分轉檔測試會跳過。

## 上傳 GitHub

將**這個專案資料夾內的檔案**上傳到自己的 repository 根目錄，保留 `.gitignore` 及 `.gitattributes`。不要改用原本含訓練資料的「Windows乾淨版」整個目錄上傳。

使用 Git 時，先檢查 `git status --short` 與 `git add --dry-run .`，確認沒有錄音、對話、金鑰、模型權重或虛擬環境，再提交。`.gitignore` 不能保護已經提交的私人檔案，公開前仍需人工確認。

本次整理不會代為建立 GitHub repository、設定遠端或推送。團隊尚未指定程式及角色圖的公開授權，因此未擅自加上 MIT 等授權。第三方模型遵循各自條款；現有來源紀錄見 `reports/provenance.md`，新增模型前應再檢核競賽規定。
