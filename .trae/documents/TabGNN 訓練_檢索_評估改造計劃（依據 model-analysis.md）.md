## 目標
- 落地 `model-analysis.md` 的建議：修正投影層用法、歸一化選型、優化器/調度器與訓練流程。
- 提升檢索效率：預計 Recall/MRR 提升 5–15%，整體計算加速 10–20%。
- 保持介面一致與可重現：統一超參、檢查點命名、隨機種子；補齊單元測試與評估/可視化/基準。

## 現狀關鍵點（代碼錨點）
- 訓練入口與核心：`train_model.py:82`（`main()`），設備設定於`84`；模型類 `DiffusionModel`：`34–75`；投影頭含 `LayerNorm+MLP+Dropout`：`49–56`；`F.normalize`：`64`。
- 超參硬編碼：`hidden_channels=256`（`137`）、`lr=5.5e-4`（`138`）、`NUM_EPOCHS=300`（`139`）、`BATCH_SIZE=128`（`141`）；`AdamW(weight_decay=1.5e-2)`（`152`）；`CosineAnnealingLR` 在 warmup 後啟用（`153–155`, `208–211`）。
- AMP/縮放：`autocast`（`182`）、`GradScaler`（`155`, `193–198`）。
- 記錄/評估：epoch 日誌（`212`）；`Recall@1/5/10`/`MRR`（`214–239`）。
- 檢查點：固定檔名覆寫 `diffusion_model_epoch.pt`（`242–244`）。
- 檢索：`retrieval.py` 入口 `main()`（`153–188`）、`retrieve(...)`（`77–151`）；相似度=點積（`139–144`）；無快取/多執行緒；`hidden_channels=128`（`110–117`）與訓練不一致；預設讀取 `diffusion_model_epoch_30.pt`（`77`）。
- 評估：`evaluate_retrieval.py` 解析與批量計算（`66–96`, `159–163`）；指標含 `Recall/MRR/MAP/NDCG`（`99–137`, `176–183`）；無可視化/輸出檔。

## 一、train_model.py 改造
- 架構與表示
  - 移除升維到 384 與投影層中的 `LayerNorm`；保留 `GNN` 輸出 256 維作為最終表示（下游使用）。
  - 新投影頭：`Linear(256→128) + ReLU`，僅用於對比損失；在投影輸出後做 `L2` 歸一化（InfoNCE）。
  - 歸一化選型：若需特徵歸一化，優先 `PairNorm`（簡易函式）或 `InstanceNorm1d` 應用於節點特徵，避免全向量 `LayerNorm` 扭曲距離結構。
- 優化與調度
  - 優化器：`Adam(lr=1e-3, eps=1e-7)`；`weight_decay` 調整到 `1e-2`（與分析一致），可參數化。
  - 調度器：從訓練開始使用 `CosineAnnealingLR(T_max=NUM_EPOCHS, eta_min=1e-5)`；可選 10% 線性 warmup（`LambdaLR` 鏈接）。
  - 取消 `autocast/GradScaler`，全程 `float32`；保留梯度裁剪 `max_norm=1.0`。
- 監控與日誌
  - 每 epoch 記錄：`train_loss`、`top1/top5 命中率`、`MRR@k`、`F1@k`（以二值 Hit@k 定義 P/R/F1），可選 `val_*`。
  - 輸出到 `stdout` + `CSV(metrics.csv)`；可選 `TensorBoard`（若環境可用）。
- 早停與檢查點
  - 新增 `EarlyStopping(patience, monitor='val_mrr', mode='max')`；每次提升即保存 `model_best.pt`，同時保留 `model_last_epoch{N}.pt`（含優化器/調度器狀態）。
  - 檢查點命名統一，`retrieval/evaluate` 皆以 `model_best.pt` 為預設。
- 批次與資料
  - 保留現有手動分批，或可選切換 `DataLoader`（若引入驗證集）。
  - 隨機種子：設定 `torch`, `numpy`, `random`，並固定 `torch.backends.cudnn.deterministic`/`benchmark=False`。
- 正則與增強
  - 增加嵌入 Dropout（投影前 `p=0.1–0.3` 可調）；可選高斯噪聲注入（小幅 `σ=0.01`）。
  - Dropout 比率與 `weight_decay` 可透過 CLI 調整。

## 二、retrieval.py 改造
- 計算效率
  - 預先計算並快取全表格嵌入：`torch.save('cache/table_emb_<modelhash>.pt')`；載入時校驗模型檢查點路徑/哈希。
  - 批量查詢：`SentenceTransformer.encode(batch, batch_size=...)`，向量化 `torch.mm(Q, T^T)` 再 `topk`；可選 `num_threads`（`torch.set_num_threads`）。
  - 多進程選項：當查詢極多時，用 `multiprocessing` 將批次分塊後合併結果（避免外部依賴）。
- 相似度方法
  - 提供 `--metric {cosine, dot, l2}`；`cosine` 對應先 `normalize` 再 `dot`（預設）。
- 鄰居節點採樣
  - 計算全表格嵌入時可選 `NeighborLoader`（例如 `[15,10,5]`）以降低大圖前向記憶與時間成本；確保與 `train_model.py` 前向一致。
- 特徵重要性
  - 新增 `compute_feature_importance(method='permutation', top_k=...)`：對維度/欄位做遮蔽/擾動，度量對 `score`/命中率的影響，輸出 CSV（維度重要性排序）。
- 介面與預設
  - 統一 `hidden_channels=256` 與 `proj_dim=128`；預設模型路徑 `model_best.pt`；新增 CLI 參數（或函式參數）以覆寫。

## 三、evaluate_retrieval.py 擴展
- 指標擴充
  - 已有：`Recall@k`, `MRR`, `MAP@k`, `NDCG@k`；新增：`Precision@k`, `F1@k`, `HitRate@k`。
- 可視化
  - 生成並保存：`Recall/Precision/F1 vs k` 曲線、`MRR`/`MAP` 條形圖、分數分布直方圖；輸出至 `--out_dir`（PNG）。
- 參數對比
  - 支援 `--config_grid grid.json`（多組 `{metric, top_k, proj_dim, dropout, weight_decay}`）；迭代評估，彙總表格到 CSV/JSON。
- 結果導出
  - 追加 `--export_csv results.csv`, `--export_json results.json`；保留原有 CLI（`203–209`）。
- 一致化
  - 載入 `retrieval.DiffusionModel` 的超參與訓練一致（`hidden_channels=256`）；預先計算/載入快取的表格嵌入（避免重算）。

## 四、介面一致性與可重現
- 超參：統一 `hidden_channels=256`、`proj_dim=128`、`lr=1e-3`、`weight_decay=1e-2`、`dropout=0.1–0.3`（可 CLI 覆寫）。
- 檢查點：統一 `model_best.pt`/`model_last_epoch{N}.pt`；`retrieval/evaluate` 以 `model_best.pt` 為預設。
- 隨機種子：三個腳本皆設置；記錄到日誌與結果文件。
- 嵌入器與裝置：提供 `--device {cuda,cpu}`；檢索默認跟隨模型設備；批次大小與線程數可配置。

## 五、單元測試（unittest）
- `tests/test_training.py`：小型合成圖上跑 5–10 epoch，驗證 `loss` 下降、早停觸發、最佳檢查點存在。
- `tests/test_retrieval.py`：快取文件生成與命名一致性；批量查詢與單查詢結果一致；`metric` 切換行為正確。
- `tests/test_metrics.py`：`Precision/Recall/F1/MRR/MAP/NDCG` 對已知排名的正確性。
- `tests/test_feature_importance.py`：遮蔽法對分數影響方向合理（重要特徵影響更大）。

## 六、文檔更新
- 在 `model-analysis.md` 末尾新增「落地修改與配置」章節：
  - 說明新超參與預設/CLI；早停與檢查點；快取與多進程；評估與可視化用法；基準操作與目標。
- 變更日誌：列出影響介面一致性的改動（`hidden_channels/proj_dim/checkpoint`）。

## 七、基準與目標
- 基準協議：用同一 `graph.pt` 與 `query_file`，以 `evaluate_retrieval.py` 的擴展模式跑 `k∈{1,5,10,50}`，輸出 CSV/JSON。
- 目標達成：相較改造前，`Recall@10` 與 `MRR` 提升 ≥5%（分析預期 5–15%）；若提供 `--baseline_json`，自動比較與標記達成情況。

## 風險與回退
- 若環境無 `matplotlib`，可視化步驟降級為僅輸出 CSV/JSON。
- 大圖場合若 `NeighborLoader` 不可用，改為分批前向（依節點分塊）。

## 實施順序
1) 統一模型架構與訓練流程（移除 AMP/LayerNorm、加投影/InfoNCE、調度/早停/日誌/種子）。
2) 優化 `retrieval.py`（快取、批量、metric 選項、鄰居採樣、重要性分析）。
3) 擴展 `evaluate_retrieval.py`（指標、可視化、參數對比、導出）。
4) 單元測試與文檔更新；跑基準與產出結果。