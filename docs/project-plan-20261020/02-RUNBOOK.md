# 詳細執行與協作手冊

本手冊分「現在可以使用的既有入口」與「計畫新增的開發步驟」。本次文件工作沒有替兩台電腦重裝環境或證明效能達標。所有命令從指定資料夾執行，看到錯誤就保留第一個錯誤，不反覆重裝或刪除私人資料。

## 一 從 GitHub 取得正確目錄

先安裝 Git，再於自己存放專案的父資料夾執行。若已clone，先用git status確認自己的修改；不要覆蓋原目錄。

```text
git clone https://github.com/Yun05617/future.git
cd future
git switch main
git pull --ff-only origin main
cd voice-identity-emotion-chatbot
```

計畫文件位於repo根目錄docs/project-plan-20261020/，不是應用內原有docs。PR未合併前，需取得相應PR分支才看得到新文件。

## 二 事前安裝與設備紀錄

主程式預計Python 3.12 64位元，另需FFmpeg、Ollama及瀏覽器。Windows安裝Python時加入PATH；FFmpeg須包含ffmpeg及ffprobe並加入PATH。Mac可使用官方Python與套件管理器安裝FFmpeg，勿沿用Windows的.venv。GitHub Pages只能展示靜態說明，無法直接執行本機Python與GPU模型。

Windows驗收機：RTX 4060 Ti、16GB專用VRAM、1TB儲存；CPU、系統RAM、OS及驅動待記。Mac驗收機：MacBook Air M3、16GB統一記憶體、512GB儲存；OS待記。儲存容量不是可用記憶體，Mac統一記憶體也不能視為16GB獨立VRAM。

官方下載入口：Python https://www.python.org/downloads/ ，FFmpeg https://ffmpeg.org/download.html ，Ollama https://ollama.com/download ，PyTorch https://pytorch.org/get-started/locally/ 。版本由實際成功安裝結果鎖定；不隨意把主程式舊Torch更新成其他環境要求的版本。

### 2.1 Windows檢查

於PowerShell執行。nvidia-smi若不可用，先查驅動而不是直接認定CPU推論。

```powershell
py -3.12 --version
git --version
ffmpeg -version
ffprobe -version
ollama --version
nvidia-smi
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

### 2.2 macOS檢查

```bash
python3.12 --version
git --version
ffmpeg -version
ffprobe -version
ollama --version
system_profiler SPHardwareDataType
```

硬體輸出供本機整理，發布前只保留晶片、RAM與OS，不公開序號與裝置識別碼。

## 三 現有Windows版本的安裝與兩頁同開

以下從future/voice-identity-emotion-chatbot/執行。現有安裝腳本會要求Python3.12，下載requirements列出的套件；本次未重新驗證所有套件可取得。遇到No matching distribution先確認版本、平台與Python，建立Issue記錄，不任意取消全部版本限制。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
ollama list
```

目前程式在showcase_chat.py及personal_demo_ui.py指定gemma4:12b，因此「現在啟動」若沿用此版本，需要下載同名模型：

```powershell
ollama pull gemma4:12b
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_windows.ps1
```

這是對現況的操作說明，不代表計畫已採12B作最終模型。先前約定Gemma3 4B與目前程式不一致；只執行ollama pull gemma3:4b並不會改變程式呼叫。第一週將模型名集中設定，改動兩個呼叫者與測試，經審查後才下載新模型並比較品質／速度。

啟動器同時開訓練後台 http://127.0.0.1:7860/ 與展示 http://127.0.0.1:7862/ 。保留終端機；Ctrl+C停止。若Ollama未啟動，先從應用程式啟動；需要命令服務時可在另一視窗執行ollama serve，但不要重開已占用11434的服務。

## 四 遠端版本在macOS的手動入口

此段按目前遠端Python入口整理，仍需在M3驗收；遠端原生語音回覆目前以Windows介接為主，Mac語音需移植本機版或補上後端。不可因頁面可開就標記影音功能全部相容。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
ollama list
ollama pull gemma4:12b
export PYTHONPATH="$PWD/src"
export VOICE_PROJECT_WORKSPACE="$PWD/workspace"
.venv/bin/python scripts/run_competition_demo.py
```

同一命令啟動兩個頁面。若使用Mac本機舊開發目錄而非GitHub clone，既有platform/macOS/啟動全部.command是另一份檔案結構，不能在遠端版本假裝存在。先比對後以獨立PR移植，並讓Windows與Mac共用src，避免複製整個專案成兩套。

安裝後分平台檢查推論裝置，不依顯示卡型號推定每個模型已用GPU：

```bash
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'
```

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## 五 首次建立資料與訓練

步驟1，在訓練後台依成員保存5–10段自然聲音，每段約3–10秒，避免所有人講完全相同背景音或只背同一句。這是收集起點，不是已證明足量。保存後按建立／更新聲紋範本；它是用預訓練特徵建立範本，不是從零訓練完整聲紋網路。

步驟2，按正面、負面、平靜分類加入情緒訓練錄音，再訓練三類分類器。每類至少兩段僅是程式最低條件。標籤由錄音者確認，不因內容含「難過」就當作聲音標籤。

步驟3，隔天另錄新句子作封存測試，不能拿來調門檻或訓練。調參使用另外的驗證錄音；若看過測試結果才改模型，原資料降為驗證集，重新錄正式測試。

步驟4，用兩段明顯不同的新錄音測重錄流程，確認request_id、音訊hash、逐字稿不同，避免舊音訊重送。確認四類baseline與三類展示資料、模型及指標沒有混用。

步驟5，PIN與記憶功能開發完成後才用私人正式資料試家庭隔離。程式未有PIN時，不將現有聲紋候選稱為安全登入。

## 六 A線照片說話執行順序

第1步：開feature/avatar-spike分支，在獨立目錄及虛擬環境試選定後端，記錄來源及模型權重；不要把研究模型的requirements直接安裝進聊天.venv。

第2步：固定一張有使用權的正面照片與10秒合成中文語音，先做命令列輸入至MP4輸出；再以20及30秒完整音檔測試兩平台。失敗時記錄是哪個階段、缺的依賴或MPS算子，不把失敗時間從統計移除。

第3步：加入AvatarJob契約與假後端，讓B線不需等待大模型即可串接。真後端沿用job_id，完成後先寫暫存檔、檢查音軌及長度，再原子改名為可播放結果。

第4步：照片前處理只做一次並可快取；每輪用實際TTS音檔，不重複合成第二份不同語速的音軌。若模型只輸出無聲影片，用FFmpeg合成原音檔。以下是已存在影像／音訊的通用合成步驟，不會自動產生嘴型：

```bash
ffmpeg -i rendered_silent.mp4 -i reply.wav -map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart reply.mp4
ffprobe -v error -show_entries format=duration -show_entries stream=codec_type,codec_name,duration -of json reply.mp4
```

不要用-shortest掩蓋長度不一致；先確認rendered_silent與reply時長及時間軸。真人自然度、嘴型同步、頭部動作需另外評審，影片有音軌不等於已通過。

第5步：測30秒提示、改聽語音、不重播、取消、重新錄音、換人、鎖定及刪照片。使用者更換照片會增加avatar_revision，舊工作不能覆蓋新形象。

## 七 B線核心與記憶執行順序

先集中模型設定，保留已知可跑的路徑；接著寫session與權限檢查，再接PIN與前台，不採單純隱藏HTML來保護資料。PIN建議採經審查的密碼雜湊與每人salt，含錯誤次數限制及安全重設；數字PIN長度及復原流程需先決策。不要將PIN置於URL、log或Git。

將speaker候選與authenticated_user分離；session換人後增加generation。非同步轉寫、TTS、人像及記憶結果只可回到相同user與generation。低信心訪客不讀寫成員記憶。文字對話也不能繞過授權。

完成最小歷史查詢後才加入長期記憶候選、本人勾選確認與刪除預覽。每個功能要先有越權及失敗測試，再接真資料。最後加入只在明顯正負相反時確認的情緒規則，區分聲音估計與本人說法。

## 八 日常GitHub同步

本專題更新同步到Yun05617/future。分支建立在對的repo根目錄，不在含所有桌面檔案的上層。更新前讀取main與自己的修改；不要強推、不要將別人的未提交變更當作自己的變更。

```text
git status --short
git switch main
git pull --ff-only origin main
git switch -c feature/avatar-jobs
```

只加入自己這次的具體檔案，例如契約文件。先確認檔案存在再執行git add：

```text
git add docs/project-plan-20261020/03-INTEGRATION-CONTRACT.md
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "docs: clarify avatar job integration contract"
git push -u origin feature/avatar-jobs
```

在GitHub建立PR，附問題、行為、測試與未驗證事項，另一位開發者審查後再合併。main有更新，兩人各自pull；衝突時先保留兩方變更意圖，不整檔覆蓋。新分支命名例feature/session-pin、feature/memory-review、feature/avatar-jobs、fix/audio-reuse。每次提交同步更新對應README、契約、測試及模型來源。

本地Git提交與遠端PR不是備份私人資料的方法。確認根目錄.gitignore存在且生效；已被追蹤的私人檔案不受ignore保護，要另行處理Git歷史與撤銷金鑰。本手冊不指示上傳整個桌面或本機資料。

## 九 常見失敗與停止條件

連接埠占用：先確認原服務是否還在；不任意結束不明行程。模型不存在：比對程式指定名稱與ollama list。環境衝突：記錄原.venv及套件清單，在隔離環境重現。GPU未使用：逐模型確認device，不以Ollama用了GPU就假定Whisper也用了。30秒失敗：分解TTS、排程、渲染、瀏覽器時間，記錄實際回答長度，不直接增大timeout假裝改善。

PIN或記憶隔離失敗時，停止使用私人資料；影片生成失敗可降級語音，不影響安全狀態。不要自動刪除workspace、data、models或本機資料來排除問題。清理暫存只刪已標示為工作輸出且不被當前播放使用的檔案。
