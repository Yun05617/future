# 測試表使用方式

CSV以UTF-8編碼；一列一輪或一段，不預填結果。用member_a等匿名ID，真實音訊路徑及私人資料表僅保留本機。模板可上傳，填寫結果先去識別再分享。

- hardware.csv：每台驗收機一列，補CPU、系統RAM、專用VRAM、版本及commit；不公開硬體序號。
- identity.csv：每音檔每模型設定一列，outcome填correct/wrong_member/guest/error；invalid另記原因。
- latency.csv：每平台每輪一列，mode填voice或avatar；wall_seconds含人工等待，manual_wait_seconds另列。各stage可能平行，不相加當總時間。
- reply_preference.csv：每人10組；left_version/right_version由主持人保密，choice填left/right/tie；必須保留40組完整分母。
- avatar.csv：回答音檔長度、解析度、同步、自然度及降級選擇；逾時不算30秒內成功。
- issues.csv：執行backlog與缺陷，一次一項，記相依、重現步驟、驗收及PR。

原始測試資料不放本目錄；複製到被Git忽略的本機資料夾再填寫。模擬或單元測試和實機測速分開標記。
