# EchoMimic Mac 實驗性移植

上游 https://github.com/antgroup/echomimic ，固定提交 `c32b3a557003f84ead1483a2d2386035685d984d`。此目錄只保存移植補丁與測試環境，不包含模型、範例媒體或個人資料。上游程式、模型仍適用各自授權，本補丁不改變其授權。

## 重建

1. 安裝 Python 3.10、uv、FFmpeg，確認 ffmpeg 在 PATH。
2. clone 上游並 checkout 上述提交。
3. 在上游根目錄執行 `git apply /此目錄/mac-port.patch`。
4. 將 overlay 內的 mac 與 configs 內容複製到上游根目錄。
5. 依上游說明下載音訊驅動權重到 pretrained_weights；外接磁碟可使用符號連結。權重來源 https://huggingface.co/BadToBest/EchoMimic 。
6. 執行 `uv venv --python 3.10 .venv`、`uv pip install --python .venv/bin/python -r mac/requirements.txt`。
7. 執行 `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python mac/check_device.py`。
8. 執行 `bash mac/run.sh -L 12 --steps 2`，輸出位於 output。請先閱讀驗證報告的限制。

已在 M3 MacBook Air 16 GB 完成短片生成及獨立音訊合成驗證，並非官方 Mac 支援，也不是完整對話產品。2026-11-05 已以使用者照片完成修正後的完整入口；Windows/CUDA 入口不屬於此次驗證範圍。

## 照片與音訊上傳頁

將 overlay 內檔案依相對路徑放進上游專案，安裝更新後 mac/requirements.txt，執行 `bash mac/start_test_ui.sh`，開啟 http://127.0.0.1:7881 。支援照片及 0.5～5 秒音訊、進度、取消、MP4 預覽與下載。每次仍需重新載入模型，不是即時生成。詳見 overlay/mac/照片測試平台使用說明.md。

驗證：6 項輸入／程序測試通過；首頁 HTTP 200 並已在瀏覽器顯示；空輸入提示正常；上傳官方樣本與 1 秒音訊後建立獨立工作並啟動真實推論程序，取消後程序停止。本輪未透過新介面等候完整影片生成。Gradio 固定搭配 FastAPI 0.103.2、Starlette 0.27.0、Pydantic 2.4.2，避免舊介面與新模板 API 不相容。

## 完整操作手冊與最新實測

請閱讀[照片說話專案：技術、部署與操作手冊](照片說話專案_技術與操作手冊.md)，包含環境重建、啟動、權重清單、使用步驟、排錯、整合與提示詞。

2026-11-05：透過平台後端以真實照片與 2.1105 秒合成語音完成 10 步生成，全程約 31 分 55 秒。8 項測試通過；生成影片可完整解碼，影音皆從零開始，長度差約 28 毫秒。尚未量化評估音素嘴型同步，也未達成即時生成。詳細紀錄見手冊第 10、11 章；個人照片、音訊及影片不包含在儲存庫。
