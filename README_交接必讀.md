# 小組交接版：2026-11-09

本版保留程式、訓練資料及已訓練模型，供組員接續開發。原電腦資料不受本次整理影響。

## 兩個檔案包

1. **小組交接_程式與訓練資料_2026-11-09.zip**：程式、身分錄音、情緒錄音、身分與情緒分類器、特徵快取、歷史模型備份、擴增來源及 FFmpeg。
2. **小組交接_預訓練模型包_2026-11-09.zip**：現有 XLS-R、ECAPA-TDNN、Whisper small 與 Gemma 4 12B 權重，約 10.4 GiB。這是可選的離線權重包，省去重抓模型；Python 套件與 Ollama 主程式仍須安裝。

將兩包解壓縮到同一個上層資料夾，合併其中的 `voice-identity-emotion-chatbot/`。第一包可單獨使用，有網路時會下載需要的語音模型，Gemma 可用 `ollama pull gemma4:12b` 取得。

模型包內沒有 Ollama 帳號、SSH 金鑰、API 金鑰或其他使用者設定。大型權重不要直接提交到一般 Git 儲存庫。

## 組員第一次使用

1. 安裝 Python 3.12（64 位元）與 Ollama。
2. 解壓縮程式包，雙擊 `setup_windows.bat` 建立自己的 `.venv`。不要沿用別台電腦的虛擬環境。
3. 如有模型包，解壓縮合併到同一專案資料夾。退出 Ollama 後，雙擊 `匯入本機Gemma模型.bat`，再開啟 Ollama，執行 `ollama list` 確認 `gemma4:12b`。沒有模型包則執行 `ollama pull gemma4:12b`。
4. 雙擊 `啟動全部_Windows.bat`。啟動器會修正複製的訓練 JSON 路徑，音檔與分類器權重不改寫，也不會自動重新訓練。
5. 保持啟動視窗開著。前台 http://127.0.0.1:7862/；後台 http://127.0.0.1:7860/；擴增 http://127.0.0.1:7876/。

已附 `tools/ffmpeg.exe`，啟動器會加入當次程序 PATH。首次匯入 Gemma 會複製約 7.5 GiB 權重到使用者的 `.ollama/models/`，請保留足夠磁碟空間；既有同內容檔案會略過，不覆寫不同內容。

## 保留哪些訓練狀態

| 內容 | 路徑 |
| --- | --- |
| 正式身分訓練錄音 | `workspace/01_原始音檔/身分辨識/` |
| 已訓練身分分類器 | `workspace/03_語音身分辨識/learned_identity.npz` |
| 身分特徵與模型備份 | `workspace/03_語音身分辨識/` |
| 情緒訓練錄音 | `data/raw/personal_demo/` |
| 已訓練情緒分類器 | `models/personal_demo.joblib` |
| 情緒訓練特徵與報告 | `data/processed/personal_demo/` |
| 擴增來源與輸出 | `workspace/09_資料擴增/` |

此刻正式身分錄音：莊媗丞 320、姜昀廷 320、紀律廷 320、趙科翰 319，共 1,279 筆；這是檔案數，包含擴增副本，不是獨立原始錄音數。模型取用的有效樣本數以模型訓練摘要為準。

身分辨識使用分類器支持度，未四捨五入數值達 80% 才顯示姓名。前台信心度顯示分類器支持度至兩位小數；聲紋餘弦相似度不作第二道門檻。聲紋及情緒的分類器分開，不會互相訓練。

聊天紀錄、聊天原音、逐字稿、完整對話分析、競賽錄影及測試輸出沒有交接。身分「處理後音檔」是中間產物，也不附帶；先前要求刪除但受權限阻擋的 651 個舊檔不納入。正式身分訓練音檔及現有模型不受影響。

## GitHub 上傳

本版 `.gitignore` 已允許提交正式訓練資料及分類器。建議使用小組的私有 repository，以 Git／GitHub Desktop 上傳資料夾內檔案，包含 `.gitignore`。

FFmpeg 約 84 MiB，超過網頁單檔 25 MiB 上傳限制，可透過 Git 提交或另外傳給組員。大型預訓練模型包超過一般 Git 單檔 100 MiB 限制，請另行傳輸，不要直接加入 Git 歷史。[GitHub 官方說明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

這份版本未在全新電腦重新安裝全部套件；使用現有相依環境測試，組員換電腦後仍需確認錄音權限、Ollama 與語音播放。
