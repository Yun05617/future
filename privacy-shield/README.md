# Privacy Shield

基於 LSB 數位隱寫術的影像隱私保護展示系統，使用 Python 與 Tkinter 提供操作介面。

## 功能

- 使用 1–5 張圖片模擬電腦日常操作，可手動切換或輪播。
- 未授權擷取時，將目前畫面嵌入隨機名畫的 RGB 低位元，輸出 PNG 藏圖。
- SHA-256 從密碼產生亂數種子，NumPy 依此排列與還原影像資料。
- 輸入展示密碼 `family123` 後，可解碼前一次藏圖或擷取正常畫面。
- 錄影展示輸出逐格 PNG 藏圖及分鏡圖。
- 保留命令列的 encode、decode、demo 與 metrics 功能。

## 系統範圍

目前是「應用程式內的擷取流程模擬」。程式沒有攔截 Windows 截圖 API、偵測外部駭客或保護整個桌面。最小化後程式仍開著，但不會因此阻止其他軟體截圖。錄影功能產生影格與分鏡圖，沒有輸出 MP4。

SHA-256 在此用於產生排列種子，不能視為完整加密或身分驗證機制；展示密碼寫在程式內。解码是有損還原：預設保留原圖每個色彩通道的高 4 位元，並將原圖縮放至封面尺寸。`metrics` 使用各色彩通道的全圖統計式 SSIM，不是滑動視窗 SSIM。

## 環境與啟動

建議使用 Python 3.12 或 3.13。Windows 的 Python 安裝需包含 Tcl/Tk；Tkinter 隨 Python 提供，不需要透過 pip 安裝。

在此資料夾開啟 VS Code 終端機，逐行執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

VS Code 按 `Ctrl+Shift+P`，選擇 `Python: Select Interpreter`，指定 `.venv\Scripts\python.exe`。打開 `app.py` 按「Run Python File」即可顯示介面，不需輸入參數。

Anaconda / Spyder 使用者應在 Spyder 所使用的環境安裝 `requirements.txt`，再開啟 `app.py` 執行。資料路徑以程式位置為準，不受終端機目前目錄影響。

## 資料夾

```text
privacy-shield/
  app.py                 # GUI 與命令列入口
  requirements.txt       # Python 相依套件
  README.md
  .gitignore
  assets/
    daily/               # 本機的 1–5 日常畫面，預設不提交
    paintings/           # 本機名畫素材，預設不提交
    samples/             # 可隨專案提交的程式繪製範例
  outputs/               # 擷取、解碼與錄影結果，預設不提交
```

日常畫面放在 `assets/daily/`，檔名使用 `1.png` 到 `5.png`，也支援 `.jpg`、`.jpeg`、`.bmp`。名畫放在 `assets/paintings/`，目前辨識的檔名列在 `app.py` 的 `ART_FILE_NAMES`；換圖時更新此清單即可。每次修改圖片後重新啟動程式。

本機素材已從原資料夾複製過來。Git 會忽略日常截圖及外部名畫；沒有這些素材的下載者會自動使用 `assets/samples/` 的兩張範例圖片。確認截圖可公開、名畫授權適用後，才調整 `.gitignore` 決定公開哪些素材。範例來源說明見 [assets/SOURCES.md](assets/SOURCES.md)。

## 展示步驟

1. 執行 `app.py`，切換日常畫面或啟動輪播。
2. 按「模擬駭客截圖」，在右側檢視名畫藏圖。
3. 輸入 `family123`，按「解鎖」，檢視還原畫面。
4. 在授權狀態再按截圖或錄影，檢視正常畫面。
5. 按「重新鎖定」恢復未授權模式。

所有 GUI 輸出存入 `outputs/`，同名結果會覆寫。既有程式也會在模擬截圖時建立解碼證明檔；輸出資料夾並非安全儲存區。

## 命令列驗證

以下使用已安裝相依套件的 Python：

```powershell
python app.py demo --key family123
python app.py encode --cover assets/samples/cover_mona_style.png --secret assets/samples/secret_fall_scene.png --output outputs/stego.png --key family123
python app.py decode --stego outputs/stego.png --output outputs/recovered.png --key family123
python app.py metrics --cover assets/samples/cover_mona_style.png --stego outputs/stego.png --secret assets/samples/secret_fall_scene.png --recovered outputs/recovered.png
```

藏圖請維持 PNG 格式；轉為 JPEG 或縮放後可能破壞低位元資料。自訂命令列輸出路徑時，先建立父資料夾。

## 上傳 GitHub

先在 GitHub 建立空的 repository，不要勾選初始化 README。然後在 `privacy-shield` 資料夾逐行執行：

```powershell
git init
git add .
git status --short
git commit -m "Initial Privacy Shield demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/privacy-shield.git
git push -u origin main
```

將 `YOUR_USERNAME` 與 repository 名稱替換成實際值。提交前以 `git status --short` 確認清單；原企畫書、簡報、舊程式與 demo_output.zip 都不在此專案資料夾內。

本專案尚未指定開源授權。公開 repository 不等於自動授權他人重用；可依小組及課程要求另行加入 LICENSE。
