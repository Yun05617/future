# 核心語音與照片說話整合契約

狀態：設計提案，待兩位開發者第一週實作確認。下列方法及資料結構不是現有已部署API。本版優先同機呼叫或隔離子行程；如需HTTP，僅綁127.0.0.1並驗證工作授權，不開分享網址。

## 一 共同責任

B線輸出完成的reply_text與本機WAV；A線以該音訊生成單支MP4。兩邊透過固定ID、工作狀態及錯誤碼溝通，禁止A線自行讀成員聊天資料。顯示層只播放當前有效session的結果。

所有ID由程式生成，不使用真名作檔名或URL。路徑只允許在指定工作區內，拒絕../、任意外部路徑、符號連結越界與遠端URL。工作檔不能放在未授權可讀的Gradio公開靜態目錄；用工作token或後端授權串流取用。刪除與鎖定後，舊URL不得繼續可讀。

## 二 Session資料與狀態

Session包含session_id、authenticated_user_id或null、generation整數、last_activity_at、expires_at、camera_consent及mode。PIN驗證成功後才指派authenticated_user_id；前端傳入的owner_id不能作授權依據。

狀態轉移：LOCKED → CANDIDATE → PIN_REQUIRED → AUTHENTICATED。AUTHENTICATED遇換人、逾時或手動鎖定回LOCKED；低信心進GUEST並清除舊授權。GUEST不能查成員記憶。舊工作在鎖定時撤銷而不是借用新token繼續展示。

建議介面：verify_pin(session_id, candidate_id, pin)、lock_session(session_id)、end_conversation(session_id)。每次解鎖或鎖定增加generation；後端不信任畫面目前顯示的人名。PIN欄位不得出現在一般序列化log。

## 三 ReplyEnvelope範例

```json
{
  "schema_version": "1.0",
  "request_id": "req_demo_001",
  "session_id": "session_demo_001",
  "session_generation": 4,
  "owner_id": "member_a",
  "conversation_id": "conv_demo_001",
  "message_id": "msg_demo_002",
  "reply_text": "今天聽起來很忙，想先說哪件事？",
  "audio_asset_id": "audio_demo_001",
  "audio_duration_ms": 12500,
  "audio_sha256": "在執行時填入真實64位十六進位雜湊",
  "assistant_tone": "supportive",
  "speech_emotion": "negative",
  "confirmed_feeling": null,
  "created_at": "2026-10-20T08:00:00Z"
}
```

此範例用虛構ID。schema_version改動需兼容或更新契約。confirmed_feeling為可選的本人描述，不強迫對應三類情緒。audio_duration_ms來自讀取音檔，不能以字數估算替代。avatar可用情緒提示選動作，但不接觸完整歷史。

## 四 AvatarJob與可抽換後端

建議方法create_job(reply_envelope, avatar_id, avatar_revision, output_profile)、get_job(job_id)、cancel_job(job_id)、get_authorized_media(job_id, session_token)。output_profile包括解析度、FPS、編碼及品質參數，先以技術測試決定，不先承諾1080p。

工作資料最少包含job_id、request_id、owner_id、session_generation、avatar_revision、audio_sha256、status、created_at、started_at、finished_at、backend_name、backend_revision、artifact_id、error_code。先假後端再真模型；用戶只看進度文字，不暴露原始本機路徑與traceback。

狀態為QUEUED、RUNNING、READY、FAILED、CANCELLED。TIMEOUT不是自動取消狀態：超過30秒先出選項，繼續等可維持RUNNING；改聽語音則另記delivery_mode=audio_fallback；取消要使job失效並撤銷media存取。不可取消GPU算子時，讓計算結束後丟棄結果，不承諾已立即釋放GPU。

同一request_id加同一session_generation及avatar_revision的重送應冪等，不重複渲染或追加聊天。新錄音必須新request_id。快取key包含照片hash、revision、音訊hash、模型版本與輸出參數，不可只以檔名reply.wav當key，防止讀到前次回答。

## 五 同步與播放

READY前檢查音軌存在、影音時長、解碼與無黑畫面。建議容忍音畫時間偏移絕對值≤150ms，屬待團隊確認的品質門檻，不是現有模型結果。播放器使用影片內唯一音軌，不能同時呼叫另一個audio播放相同回答。

照片改動、成員切換、鎖定、取消及新請求覆蓋，皆使舊結果失效。前端收到READY時再次比對request_id與session_generation，再取得授權後媒體；防止取消後的背景影片突然播出。使用者選改聽語音後，READY只提供重播按鈕，不自動啟動音軌。

## 六 錯誤碼

| 錯誤碼 | 使用者行為 | 開發者處理 |
| --- | --- | --- |
| AUTH_REQUIRED | 重新確認並輸入PIN | 不回傳私人資料 |
| SESSION_STALE | 回到目前對話 | 丟棄舊工作，撤銷舊媒體 |
| INVALID_PHOTO | 重選清楚單人照片 | 記錄無素材內容的驗證原因 |
| AUDIO_NOT_READY | 等待或重試本輪 | 不借用上輪音檔 |
| MODEL_NOT_READY | 顯示載入狀態或選語音 | 區分冷啟動及推論 |
| INSUFFICIENT_MEMORY | 改聽語音或取消 | 降低配置前先記錄峰值 |
| RENDER_FAILED | 可聽語音及看文字 | 保留錯誤階段，避免重寫聊天 |
| MEDIA_INVALID | 重新生成或改語音 | 不發布壞檔 |

## 七 長期記憶契約

MemoryCandidate欄位：candidate_id、owner_id、conversation_id、source_message_ids、proposed_text、status、created_at。status為PENDING、CONFIRMED、REJECTED或DELETED。只有本人通過PIN後可列出、修改及確認；LOCKED狀態不能在HTML預先嵌入候選文字。

MemoryItem欄位：memory_id、owner_id、text、source_conversation_ids、source_message_ids、confirmed_at、version、status。多來源關聯需正規化；刪除預覽從真實關聯查詢，不由LLM猜測。建議採SQLite交易處理更新與刪除，這是待實作選擇，不強迫替換既有所有檔案。

刪除流程：查授權 → 生成含相關記憶的預覽 → 本人選保留或刪除 → 同一交易刪聊天及更新記憶 → 更新索引並撤銷相關影音工作。檔案刪除失敗列入重試佇列，不假裝完全遺忘。保留記憶的選擇只授權保留該文字，不保留整段已刪除逐字稿。

## 八 離線與資源

所有推論服務預設本機；初次套件及模型下載需要網路。預載模型、照片快取及音檔預處理需遵守記憶體預算；Mac同時載多個大模型可能換頁。Ollama keep_alive可減少重載，但不能改善影片本身的運算量，也不能保證整條流程10秒或30秒。

## 九 第一週必備契約測試

固定虛構ReplyEnvelope即可測假影片工作；檢查冪等、取消、換人、過期token、路徑越界、錯照片revision、未就緒音檔及重播互斥。這些測試不需真實家人照片或GPU，先通過才能串接大模型。
