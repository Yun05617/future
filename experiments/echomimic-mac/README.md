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

已在 M3 MacBook Air 16 GB 完成短片生成及獨立音訊合成驗證，並非官方 Mac 支援，也不是完整對話產品。修正後完整入口仍待重跑；Windows/CUDA 入口不屬於此次驗證範圍。
