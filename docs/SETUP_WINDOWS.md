# Windows 操作說明

## 環境

Python 3.12 64 位元、FFmpeg、Ollama，以及可錄音的瀏覽器。麥克風需要瀏覽器授權；回覆播放需要喇叭或耳機及可用的 Windows 語音。

在安裝前確認：

```powershell
py -3.12 --version
ffmpeg -version
ollama --version
```

FFmpeg 需要可執行檔，不是 `.tar` 原始碼壓縮檔。本 repository 不散布 FFmpeg 執行檔；請自行安裝並加入 `PATH`，或依 `tools/README.md` 放置本機副本。

## 安裝與啟動

於專案根目錄執行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
ollama pull gemma4:12b
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_windows.ps1
```

若 Python 不在一般搜尋路徑，可指定執行檔：

```powershell
.\setup_windows.ps1 -PythonPath 'D:\Python312\python.exe'
```

只開單一頁面：

```powershell
.\start_windows.ps1 -Mode showcase
.\start_windows.ps1 -Mode training
.\start_windows.ps1 -Mode augmentation
```

預設全部啟動，包含 7860 後台、7862 前台及 7876 擴增工具。啟動器先檢查連接埠，不會終止其他程式。若出現 7860／7862／7876 已使用，先關閉原服務的終端機，再重新啟動。正常執行時不要關閉啟動視窗，停止請按 `Ctrl+C`。

## 資料位置

預設所有身分錄音、逐字稿、分析紀錄與聊天紀錄都存入本專案的 `workspace/`。情緒分類器與訓練資料另存於 `models/`、`data/raw/`、`data/processed/`；模型快取存於 `.cache/`。這些路徑均被 Git 忽略。

需要沿用自己電腦既有的研究工作區時，可以明確指定，不必把資料移進公開 repository：

```powershell
.\start_windows.ps1 -WorkspaceRoot 'D:\PrivateVoiceWorkspace'
```

如果直接執行 Python 入口，需要自行設定搜尋路徑：

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
$env:VOICE_PROJECT_WORKSPACE = Join-Path $PWD 'workspace'
.\.venv\Scripts\python.exe scripts\run_competition_demo.py
```

不建議沿用其他電腦的 `.venv`，請重新安裝。搬移個人資料或模型時，使用私下傳輸並確認說話者同意，不要將它們加入 Git。

## 身分學習與擴增

加入身分錄音後按「更新聲紋並重新訓練」。第一次擷取全部特徵較慢，後續未變動的檔案可沿用快取。品質提醒不會自動刪檔。分類器至少需要兩位成員各兩段不同檔案，這只是最低條件，不代表資料充分。介面只接受分類器支持度 ≥ 80% 的身分；未達門檻或尚無可用分類器時顯示未知使用者。前台右側的信心度即為此支持度。

擴增入口支援錄音／上傳、方法勾選、預覽與 ZIP 下載，結果另存於 `outputs/reference-augmentation/`。原始來源備份不包含在 ZIP 中，音檔不會自動進入身分或情緒資料集。相同來源及其副本不可分到訓練與測試兩邊。

## 先前四類 baseline

`data/manifests/example_manifest.csv` 只展示 `neutral`、`happy`、`angry`、`sad` 的資料格式，沒有提供實際音檔。使用真實資料前，另外建立被忽略的 `manifest.csv`；同一個 `speaker_id` 必須只屬於一個 split。

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe scripts\train.py --manifest data/manifests/manifest.csv --audio-root data --output models/baseline-1.joblib
.\.venv\Scripts\python.exe scripts\predict.py data/raw/example.wav --model models/baseline-1.joblib --utterance-id demo-001
```

## macOS

保留原有跨平台程式及相依條件，但這次整理以 Windows 啟動流程為驗證對象。macOS 可自行建立 Python 3.12 虛擬環境，安裝相依套件、FFmpeg 與 Ollama，設定 `PYTHONPATH=src` 後執行 Python 入口；Windows 語音合成介接不適用於 macOS。
