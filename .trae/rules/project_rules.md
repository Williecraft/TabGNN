# 表格檢索系統完整實作提示詞

請基於以下需求，幫我設計並實作一套完整的表格檢索系統，分為三個獨立的 Python 腳本：建圖、訓練、檢索。

---

## 系統概述

### 目標
構建一個基於異質圖神經網路（Heterogeneous GNN）+ 對比學習的高效表格檢索系統，支援大規模資料庫，並能透過自然語言查詢快速定位相關表格。

### 核心技術
- **圖結構建模**：將資料庫中的表格與欄位構建為異質圖
- **對比學習**：透過查詢-表格配對訓練，拉近正例、推遠負例
- **圖擴散檢索**：利用 APPNP 在圖上進行訊息擴散，輸出檢索分數

### 資料路徑
- 表格資料：`table/train/feta/table.jsonl`
- 查詢資料：`data/train/feta/generate_query.jsonl`（每條包含 `table_id` 和 `queries` 列表）

---

## 架構設計要求

### 檔案 1: `build_graph.py` - 圖構建與持久化

**功能**：
1. 讀取 `table/train/feta/table.jsonl` 中的所有表格資料
2. 構建異質圖結構（HeteroData）
3. 將圖保存為 `graph_cache.pt`，以便後續重複使用

**圖結構設計**：
- **節點類型**：
  - `table`：表格節點，特徵為 512 維文本 embedding（表格名稱 + 欄位名 + 前 10 筆資料）
  - `column`：欄位節點，特徵為 512 維文本 embedding（欄位名 + 前 10 筆值）

- **邊類型**：
  1. `(table, has_column, column)`：表格-欄位歸屬關係
  2. `(column, rev_has_column, table)`：反向邊
  3. `(table, sim, table)`：表格間相似度邊（基於 embedding cosine similarity，top-K=10）
  4. `(column, col_sim, column)`：欄位內容相似度邊（top-K=3）
  5. `(column, name_sim, column)`：欄位名稱相似度邊（跨表格，top-K=5）
  6. `(column, dist_sim, column)`：數值欄位分布相似度邊（基於直方圖，top-K=5）

**Embedding 生成**：
- 使用 `SentenceTransformer('distiluse-base-multilingual-cased-v2')` 生成 512 維向量
- 支援 fallback 到 hash-based embedding（當 SentenceTransformer 不可用時）

**輸出**：
- `graph_cache.pt`：包含完整的 HeteroData 物件
- 元數據：`table_ids`、`sheet_names`、`table_id_to_idx` 等映射關係

**關鍵需求**：
- 支援無限制表格數量（移除 `max_tables` 限制）
- 提供進度顯示（每處理 100 張表格輸出一次）
- 錯誤處理與日誌記錄

---

### 檔案 2: `train_model.py` - 對比學習訓練

**功能**：
1. 載入 `graph_cache.pt` 圖結構
2. 讀取 `data/train/feta/generate_query.jsonl` 查詢資料
3. 使用對比學習（InfoNCE Loss）訓練 GNN 模型
4. 保存訓練好的模型為 `retrieval_model.pt`

**模型架構**：
```python
class DiffusionRetrievalGNN(nn.Module):
    - 兩層 HeteroConv（使用 SAGEConv）
    - APPNP 擴散層（K=10, alpha=0.2）
    - 查詢投影層（512 → hidden_dim）
    - 線性分類器（hidden_dim → 1）
```

**訓練策略 - 對比學習**：

1. **正樣本構造**：
   - 從 `generate_query.jsonl` 讀取每條查詢的 `table_id`
   - 將查詢與對應表格視為正例對 `(query, table⁺)`

2. **負樣本策略**：
   - **隨機負樣本**（Epoch 1-10）：每個查詢隨機抽取 8 張非正確表格
   - **硬負樣本挖掘**（Epoch 11+）：
     ```python
     # 選擇分數最高但不正確的 K 張表格作為負樣本
     with torch.no_grad():
         scores = model(data, query_vec)
     hard_negs = torch.topk(scores, k=K+1).indices
     hard_negs = [idx for idx in hard_negs if idx != pos_idx][:K]
     ```

3. **損失函數 - InfoNCE**：
   ```python
   def contrastive_loss(scores, pos_idx, neg_idxs):
       logits = torch.cat([scores[pos_idx].unsqueeze(0), scores[neg_idxs]])
       labels = torch.zeros(1, dtype=torch.long)
       return F.cross_entropy(logits.view(1, -1), labels)
   ```

4. **前向傳播機制**：
   - 查詢注入方式：計算查詢向量與所有表格 embedding 的相似度，作為擴散初始權重
   - 不在訓練時直接注入到正確表格節點（避免訓練-評估不一致）

**訓練配置**：
- Epochs: 50（可透過環境變數調整）
- Learning Rate: 1e-3
- Optimizer: Adam
- Batch Processing: 支援分批訓練（每批 500 張表格）以節省記憶體

**評估指標**：
- Recall@5：正確表格是否在前 5 名
- MRR（Mean Reciprocal Rank）：正確表格的平均倒數排名
- 每 5 個 epoch 輸出一次評估結果

**輸出**：
- `retrieval_model.pt`：訓練好的模型權重
- `training_log.txt`：訓練過程日誌（loss、recall、MRR）

**關鍵需求**：
- 支援 GPU 加速（自動檢測 CUDA）
- 支援從 checkpoint 恢復訓練
- 提供 early stopping 機制（連續 10 個 epoch 無改善則停止）

---

### 檔案 3: `interactive_search.py` - 互動式檢索介面

**功能**：
1. 載入 `graph_cache.pt` 和 `retrieval_model.pt`
2. 提供命令列互動介面
3. 接收用戶查詢並返回最相關的表格

**使用流程**：
```bash
$ python interactive_search.py
載入圖結構... 完成（共 10000 張表格）
載入模型... 完成
========================================
表格檢索系統 v1.0
輸入查詢或 'quit' 退出
========================================

> 請問有客戶訂單相關的表格嗎？

檢索結果（Top 5）：
1. [Score: 0.89] Table_1234 | Sheet: customer_orders
2. [Score: 0.76] Table_5678 | Sheet: order_details
3. [Score: 0.68] Table_9012 | Sheet: sales_records
...

> quit
```

**檢索邏輯**：
1. 將用戶輸入編碼為 512 維查詢向量
2. 調用模型前向傳播：
   ```python
   with torch.no_grad():
       scores = model(graph_data, query_vec, start_idx=None)
   ```
3. 根據分數排序，返回 Top-K 結果

**輸出格式**：
- 顯示表格索引、分數、表格名稱（sheet_name）
- 支援環境變數 `TOPK` 控制返回數量（預設 5）

**增強功能**（可選）：
- 支援多輪查詢（保留對話歷史）
- 提供查詢建議（基於常見查詢模板）
- 輸出可視化（顯示檢索路徑或注意力權重）

---

## 通用技術要求

### 1. 依賴套件
```python
torch>=2.0.0
torch-geometric>=2.5.0
sentence-transformers>=2.2.0
numpy
```

### 2. 錯誤處理
- 所有檔案讀寫操作需加 try-except
- 模型載入失敗時提供清晰錯誤提示
- 支援 graceful degradation（如 embedding 模型不可用時使用 hash）

### 3. 配置管理
- 支援透過環境變數或配置檔調整超參數
- 關鍵參數：
  ```
  EPOCHS=50
  LR=1e-3
  HIDDEN_DIM=128
  TOPK_TABLE_SIM=10
  NEG_SAMPLES=8
  TOPK=5
  ```

### 4. 文件與註釋
- 每個函數需包含 docstring（功能、參數、返回值）
- 關鍵步驟加入中文註釋
- README.md 說明使用流程

### 5. 模組化設計
- 共用函數（如 `load_jsonl`、`get_embedder`）可提取到 `utils.py`
- 模型定義可獨立到 `models.py`

---

## 預期輸出

請為我生成以下三個完整、可執行的 Python 檔案：

1. **`build_graph.py`**
   - 完整的圖構建邏輯
   - 支援大規模資料庫
   - 包含進度顯示與錯誤處理

2. **`train_model.py`**
   - 實作對比學習訓練流程
   - 包含隨機負樣本 + 硬負樣本挖掘
   - 支援 GPU 訓練與評估指標

3. **`interactive_search.py`**
   - 簡潔的命令列互動介面
   - 清晰的檢索結果展示
   - 支援多輪查詢

### 額外檔案（可選）
- `utils.py`：共用工具函數
- `models.py`：模型定義
- `config.py`：配置管理
- `README.md`：使用說明

---

## 使用流程範例

```bash
# 步驟 1: 建立圖結構（僅需執行一次）
$ python build_graph.py
正在讀取表格資料...
已處理 100/10000 張表格...
已處理 200/10000 張表格...
...
圖構建完成！已保存至 graph_cache.pt

# 步驟 2: 訓練模型
$ python train_model.py
載入圖結構... 完成
載入查詢資料... 共 5000 條查詢
開始訓練...
Epoch 1/50 | Loss: 2.456 | Recall@5: 0.234 | MRR: 0.156
Epoch 5/50 | Loss: 1.832 | Recall@5: 0.456 | MRR: 0.312
...
訓練完成！模型已保存至 retrieval_model.pt

# 步驟 3: 互動式檢索
$ python interactive_search.py
> 查詢客戶資料
Top 5 結果：...
```

---

## 關鍵優化點

1. **記憶體效率**：支援分批構圖與訓練
2. **訓練穩定性**：硬負樣本挖掘 + 學習率調度
3. **檢索速度**：圖快取 + 模型推理優化
4. **可擴展性**：模組化設計，方便後續增強

請基於以上完整需求，生成三個高品質、可直接運行的 Python 腳本。