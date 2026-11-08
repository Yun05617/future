# 人物肖像資產

- 日期：2026-10-03。
- 工具：內建 image_gen 圖片生成工具，未使用 CLI 或第三方人物模型。
- 目前資產：`assets/companion-portrait-v3.png`，透明背景 PNG；舊肖像保留在 `companion-portrait-original.png`，參考照片調整後的有背景版本保留在 `companion-portrait-v2.png`。
- 用途：展示頁標題右側的無框靜態人物；正面、嘴部無遮擋，供未來評估語音嘴型同步。
- 人物：成年韓系女性角色，依使用者提供的照片調整臉型與五官；不是該人物的真實拍攝畫面。
- 版面：桌面人物寬 176 px，窄螢幕寬 112 px；圖片保持 4:5 比例。移除邊框、底色、陰影及內距，保留透明髮絲，CSS 讓下緣柔和消失。
- 顯示時機：初始隱藏；首次錄音或音訊上傳送出後，在等待回覆的前 2 秒淡入。之後在本次頁面開啟期間持續顯示，重新錄音、切換／清除對話或發生錯誤都不隱藏、不重播淡入；重新整理頁面才重置。
- 嘴型同步與語音播放仍待處理，見 `docs/known-issues.md`。

## 生成提示

```text
Use case: photorealistic-natural. Asset type: original virtual conversational assistant portrait for a local chat UI. Create one photorealistic studio portrait of a fictional adult Korean woman, about 26 years old, friendly and quietly confident, natural Korean-inspired styling, shoulder-length dark hair tucked away from cheeks, subtle natural makeup, soft closed-mouth smile. Head and shoulders to upper chest only, front-facing at eye level with face centered, both eyes clearly visible, lips fully visible and unobstructed, relaxed neutral pose suitable as a future talking-avatar source image. Wear a simple muted sage-green blouse with modest neckline. Soft even daylight across entire face, realistic skin texture, no harsh shadow. Plain pale cool-gray studio background. Portrait 4:5 composition, a little breathing room above head, do not crop hair or chin. Premium editorial photography, approachable expression, not glamour or idol styling, not based on any real celebrity. No hands, no full body, no jewelry covering face, no text, no logos, no watermark, no drawn border or picture frame (the UI will add its own frame). Generate a high-quality bitmap.
```

## 新版編輯提示

- v2 使用內建 image_gen：以使用者照片為臉型、五官比例參考，保持原本正面半身、鼠尾草綠上衣、柔光與清晰嘴部；不複製活動背景或手勢。
- v3 使用內建 image_gen：僅移除灰色背景，保留 v2 五官、表情、髮絲、衣服與構圖，輸出真正 RGBA 透明 PNG，無白邊、無畫框。下緣淡出由 CSS 負責，不修改人物五官。

### v3 最終提示

```text
Use case: background-extraction. Edit target: the attached updated portrait of the adult female conversational companion. Remove ONLY the gray studio background to produce a genuine transparent RGBA PNG cutout for a borderless UI companion. Preserve this exact woman's facial identity and features, expression, hair, front-facing head-and-shoulders/upper-chest composition, sage blouse, original lighting and natural skin texture. Retain realistic fine semi-transparent flyaway hairs; clean alpha edges, no gray halo, no white outline. Keep the mouth and eyes completely clear. Do not draw a frame, card, border, shadow, background color or checkerboard. Preserve the portrait 4:5 canvas composition and original full head with breathing room. Keep the shoulders and upper chest, no full body and no hands. Transparently cut around the silhouette. The app will soften the bottom chest crop with CSS; do not change the face or add new features. Deliver one photorealistic transparent portrait asset.
```
