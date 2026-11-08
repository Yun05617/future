# 身分辨識診斷與更新效率

## 已套用的程式改善

- 更新時以「來源音檔 SHA-256＋SpeechBrain 模型快取的固定 revision＋特徵擷取版本」識別特徵快取。只有新增、變更或快取損壞的錄音需要重新擷取特徵；分類器仍會使用目前全部有效特徵重新訓練，不是跳過訓練。
- 無法確認模型 revision 時停用快取；模型版本變更後不混用舊快取。辨識時也檢查已發布模型的特徵版本。
- 特徵擷取使用 torch.inference_mode，避免建立不必要的梯度圖；沒有改變聲紋特徵定義。
- 更新摘要顯示快取沿用與新擷取數量，記錄音檔長度、RMS、削波比例、相對能量活動比例及品質提醒。提醒不會自動刪除或排除檔案。
- 原子發布、舊模型備份與排他訓練鎖保留。前台與後台目前只以分類器支持度 ≥ 80% 判斷身分；餘弦相似度僅供診斷。部分隔離實驗仍使用舊版 0.40 相似度指標，不能當成目前介面的接受率。

相對能量活動使用 librosa.effects.split，不是語意 VAD。背景噪音也可能被視為活動；品質提醒不是對「是否有人說話」的可靠判決。

## 診斷指令

在專案根目錄以原有虛擬環境執行：

```powershell
.\.venv\Scripts\python.exe scripts/audit_identity_model.py --workspace "你的專題資料庫" --output outputs/identity-optimization/audit
.\.venv\Scripts\python.exe scripts/evaluate_identity_groups.py --audit outputs/identity-optimization/audit
```

預設只稽核建檔音檔，不猜測測試時間或身分。需加入指定時段單人測試時，另外提供 `--session-start`、`--session-end`（含 UTC offset 的 ISO 8601 時間）與 `--expected-speaker`（已設定成員名）。三者須一起提供，結束時間不包含在區間內。依分析 JSON 檔案最後修改時間選取，修改紀錄可能改變選取範圍。預期標籤來自操作者，不使用模型預測作正確答案，仍需人工確認。

稽核不會發布候選模型。記錄包含來源音檔、品質、可追溯擴增群組及明確標記的比較限制。原始音檔與模型均留在本地，不上傳雲端。擴增來源追查同時比對原波形與目前 afftdn=-25、16 kHz 單聲道處理後波形；不能追查所有外部加工方式。

`benchmark_identity_update.py` 會驗證稽核特徵的來源雜湊與模型 namespace，再預備衍生快取，在隔離資料庫重新訓練及比較係數，不覆寫正式模型。加上 `--exclude-traceable-synthetic` 可產生排除已知副本的隔離候選。基準資料夾已存在時拒絕覆寫，需換新輸出位置。

## 測試限制

`StratifiedGroupKFold` 將能追溯的同源錄音放在同一折；不能保證未知來源的擴增關係也被辨識。交叉驗證結果屬開發篩選，不是獨立測試成績。

訓練所涵蓋的成員與有效樣本數應以當次資料為準。不可把部分成員的封閉集合結果說成全部成員的準確率，也不可把同源擴增檔當成彼此獨立的自然樣本。

未知者代理測試將一位已知成員整組移出範本，只比較對其餘人的餘弦分數；不是實際部署的分類器，也不是真實陌生人測試，不能用來宣稱已驗證未知者拒絕能力。

## 技術來源

- SpeechBrain 擴增文件：https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/speech-augmentation.html
- 分組驗證：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

本輪沒有下載新模型，也沒有引入中國來源的模型或套件。沿用專題原有的 SpeechBrain、librosa、scikit-learn 與 PyTorch。
