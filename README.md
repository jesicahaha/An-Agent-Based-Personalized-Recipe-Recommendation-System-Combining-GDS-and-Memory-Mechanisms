# 推甄備審資料：作品與研究整理

| 資料夾 | 內容 |
|---|---|
| `An-Agent-Based-Personalized-Recipe-Recommendation-System-Combining-GDS-and-Memory-Mechanisms/` | 專題：Agent 個人化食譜推薦系統 |
| `Research_Project/` | 國科會大專生計畫：健身房契約審查系統 |
| `Artificial_Intelligence/` | 人工智慧課程：神經網路參數搜尋 |
| `operating_system/` | 作業系統課程：多行程 vs 多執行緒 |
| `Programming_Languages/` | 程式語言（一）：C++ 貪食蛇 |

---

## 1. 專題：結合 GDS 與記憶機制之 Agent 個人化食譜推薦系統

系統分為 **LLM（AI 助手）** 與 **語音烹飪** 兩大模式。成果已投稿 TANET 2026 暨 ICS；記憶機制相關研究投稿 TAAI 2026。

<img src="flow.jpg" width="600" alt="系統流程">

**核心技術**
- **知識圖譜**：以 Neo4j 建立食譜知識圖譜
- **兩階段 Hybrid RAG**：BGE 中文 Embedding + Vector Index 以 ANN 取得候選食譜 → 以候選為種子節點，在投影子圖上執行 Personalized PageRank（PPR）重排
- **食材替代**：形式化為知識圖譜連結預測，比較 TransE 與 RotatE（MRR、Hits@10），最終採用 RotatE
- **語音烹飪**：Porcupine 本地喚醒詞 → WebRTC VAD → Faster Whisper → WebSocket 推送前端 → Edge TTS 播放
- **前後端**：React + FastAPI

**專案結構**
```
.
├── neo4j/              # 知識圖譜資料處理、建置
├── back_end/           # 後端服務（FastAPI、Agent、記憶、語音模組）
├── front_end/          # 前端介面
├── KG/                 # KGE 模型訓練
├── .env_example        # 環境變數範本
├── requirements.txt    # Python 套件
└── README.md
```

**我的負責項目**
- 知識圖譜建置、ANN 召回、PPR 重排
- RotatE 食材替代模型訓練與應用
- 語音互動模組（喚醒詞、WebRTC VAD、Faster Whisper、Edge TTS）
- 前後端系統串接

---

## 2. 國科會大專生計畫：基於 Hybrid RAG 與 MOT 架構之健身房契約審查系統

解決 LLM 長文本迷失問題，獨立開發。

**方法比較**
1. 純向量檢索：搜尋全部相關資料
2. Hybrid RAG + MOT：向量相似度找出對應範本條文 → 沿人工建立的固定圖關係查找應記載事項與消保法條文 → 以 Jaccard 相似度計算子圖節點重疊，超過門檻則合併為群集

**初步結果**

| 方法 | 法條命中率 |
|---|---|
| 純向量檢索 | 12.5% |
| Hybrid RAG + MOT | 87.5%（約 7 倍） |

**後續延伸**：將法規版本與有效期間納入知識圖譜，比較時效過濾的介入時機（無過濾 / 檢索後過濾 / 檢索前投影子圖）。

---

## 3. 人工智慧：雙隱藏層神經網路與參數搜尋

- 結構 5 → 20 → H2 → 1；資料切分 6 : 2 : 2（訓練 / 驗證 / 測試）
- 第一層固定 20，第二層 H2 由 3 到 20 逐一測試；每組以 3 個 Random Seed 訓練，計算驗證準確率平均與標準差並繪製誤差線圖
- **結果**：H2 = 13 最佳，平均驗證準確率約 72.73%，標準差約 2.18%，並以最佳結構在測試集評估

---

## 4. 作業系統：多行程 vs 多執行緒登入驗證伺服器

- 以 `fork`（多行程）與 `thread`（多執行緒）分別實作伺服器端
- 模擬 10、50、100、200 個併發登入請求，以互斥鎖確保共享計數正確
- 量測登入成功次數、CPU 時間、記憶體與上下文切換次數，比較兩種架構的資源開銷

---

## 5. 程式語言（一）：C++ 主控台貪食蛇

- 使用 `windows.h` 控制游標，以 `conio.h` 的 `_kbhit()`、`_getch()` 處理鍵盤輸入
- 物件導向設計：`Snake`、`Egg`、`Wall`、`GM` 分別管理蛇、食物、牆壁與遊戲流程
- 蛇身以固定大小陣列儲存座標；碰撞偵測分為邊界、自身、食物三種
- 執行環境：Windows