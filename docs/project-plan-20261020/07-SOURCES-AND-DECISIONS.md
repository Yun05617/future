# 來源決策與待確認事項

## 一 需求依據

本版需求來自團隊2026/10/20的專題訪談：多人家庭、PIN確認、低信心訪客、情緒衝突詢問、本人勾選記憶、預設連刪、兩平台驗收，以及完整回答的人像影片。後續新確認的需求以新版本記錄，舊決策不刪除。

## 二 來源索引

查核日2026/10/20，執行前仍須核對實際上游commit及權重條款。以下用於選型限制與方法，並非全部已安裝採用。

- 現有GitHub程式：https://github.com/Yun05617/future ，檢查基準f3cd00ea62713e2ba250e8fe5048a83ff98d4ea9。
- LivePortrait官方：https://github.com/KlingAIResearch/LivePortrait 。主要包含以driving素材進行人像動作轉移；平台支援聲明不保證本專題的完整管線速度。作者資訊含Kuaishou等機構，競賽用途需審查。
- SadTalker官方：https://github.com/OpenTalker/SadTalker ，安裝：https://github.com/OpenTalker/SadTalker/blob/main/docs/install.md 。提供音訊加照片生成方向，來源及相依需審查，不預先採用。
- Wav2Lip官方：https://github.com/Rudrabha/Wav2Lip 。研究版有非商業使用限制；嘴型同步不等於自然頭部與表情生成。
- Ollama API：https://docs.ollama.com/api/generate 。stream及keep_alive等參數為工具能力，不是端到端延遲保證。
- NVIDIA規格：https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/ 。團隊已確認驗收Windows為4060 Ti 16GB。
- 競賽辦法：https://innoserve.tca.org.tw/Rules/Detail?category=College 。另依團隊已提供的2026附件五切結書核對模型來源；公開GitHub不附簽名或個資文件。對資格有疑義時詢問主辦方，不自行把非官方推論當結論。
- Python：https://www.python.org/downloads/ ，PyTorch：https://pytorch.org/get-started/locally/ ，FFmpeg：https://ffmpeg.org/download.html ，Ollama：https://ollama.com/download 。

## 三 主要決策紀錄

| 決策 | 原因 | 後果 |
| --- | --- | --- |
| 聲紋候選後PIN | 誤認可能帶出家人私事 | 授權與分類分離，文字也需解鎖 |
| 低信心訪客 | 不沿用上一人記憶 | 撤銷原session，訪客不讀寫個人記憶 |
| 結束後勾選記憶 | 避免頻繁打斷及錯記 | 需候選狀態、待確認及來源關聯 |
| 刪聊天預設連刪 | 避免刪後仍被提起 | 管理索引、摘要與影音衍生資料 |
| 人像完整回答 | 優先同步與自然感 | 人像版接受30秒目標，與10秒語音版分開 |
| A/B並行開發 | 人像與核心有不同工作及環境 | 共同契約、假後端及預留整合時間 |
| 持續同步指定repo | 確保組員拿到同版本 | 使用分支PR，私人資料排除 |

## 四 待決策清單

每項需負責人及確認日。建議在第一週結束前關閉或列阻塞，不以預設值假裝已討論完成。

- PIN長度、錯誤限制、閒置秒數與重設流程；建議重設需本機管理員程序，不能一鍵繞過。
- Windows CPU、系統RAM、OS、驅動；Mac OS及實際模型device。
- 最終LLM及模型集中設定，解決規劃Gemma3與現有Gemma4不同。
- 自然人像模型來源資格、權重授權、音訊驅動能力及Mac可用性。
- 最長正式驗收回答音訊、FPS、解析度、30秒通過比例、自然度與同步門檻。
- 聲紋建議80段、語音延遲每平台30輪、影片每平台15輪的數量是否採納。
- 原始音訊、影片快取、待確認記憶與備份的保存／刪除期限。
- 記憶整合負責者及132人時預算是否可投入；若僅80人時，縮範圍或重議時程。
- 原始碼、圖片與文件的公開授權範圍；尚未授權不得宣稱任意商用。
