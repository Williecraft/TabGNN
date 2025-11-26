#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero
import torch.optim as optim
from torch_geometric.nn import GraphNorm
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.amp import autocast, GradScaler
import json
import os
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import itertools
import copy
import csv

# ====================================================================
# A. 輔助函式 (Embedder)
# ====================================================================

def get_embedder(model_name='BAAI/bge-m3', device='cuda'):
    """
    載入並返回 SentenceTransformer 嵌入模型。
    """
    return SentenceTransformer(model_name, device=device)


# ====================================================================
# B. 核心 GNN 模型定義 (Model Definition)
# ====================================================================

class DiffusionModel(nn.Module):
    """
    異構圖 GNN 模型。
    使用 GraphSAGE 聚合節點資訊，並透過一個投影頭將 GNN 輸出的
    表格特徵轉換回原始嵌入空間，以便與查詢向量進行比較。
    """
    def __init__(self, embed_dim: int, hidden_channels: int, metadata, dropout: float = 0.2):
        """
        初始化模型架構。
        
        Args:
            embed_dim (int): 輸入特徵（SBERT 嵌入）的維度。
            hidden_channels (int): GNN 隱藏層的維度。
            metadata (tuple): PyG 異構圖的元數據 (node_types, edge_types)。
            dropout (float): 投影頭中 Dropout 的比例。
        """
        super().__init__()

        # 1. 基礎 GraphSAGE 層 (用於同構圖)
        self.sage = GraphSAGE(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=hidden_channels,
        )

        # 2. 將 GraphSAGE 轉換為異構圖版本
        # aggr='sum' 表示來自不同邊類型的訊息將被相加
        self.hetero_sage = to_hetero(self.sage, metadata, aggr='sum')

        # 3. 輸出處理層
        # 使用 GraphNorm 對 GNN 輸出的節點特徵進行歸一化
        self.norm = GraphNorm(hidden_channels)
        
        # 4. 投影頭 (Projection Head)
        # 將 GNN 的隱藏特徵（hidden_channels）映射回原始嵌入維度（embed_dim）
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, embed_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict):
        """
        模型的前向傳播。
        
        Args:
            x_dict (dict): 包含節點類型到節點特徵張量的字典。
            edge_index_dict (dict): 包含邊類型到邊索引張量的字典。
            
        Returns:
            torch.Tensor: 經過 GNN 強化和投影後的 'table' 節點嵌入，
                          並進行了 L2 歸一化。
        """
        # 1. GNN 訊息傳遞
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        
        # 2. 提取 'table' 節點的特徵
        x_table = x_dict_out['table']
        
        # 3. 歸一化與投影
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        
        # 4. L2 歸一化，使其適用於餘弦相似度計算
        return F.normalize(table_features, p=2, dim=1)

    def score_tables(self, data: HeteroData, query_vec: torch.Tensor):
        """
        [評估用] 計算單一查詢向量與圖中所有表格節點的相似度分數。
        
        Args:
            data (HeteroData): 完整的圖數據。
            query_vec (torch.Tensor): 單一的查詢嵌入向量 (shape: [1, D] or [D])。
            
        Returns:
            torch.Tensor: 查詢與所有表格的相似度分數 (shape: [N_tables])。
        """
        # 1. 獲取經過 GNN 強化後的表格特徵
        # .forward() 會返回 L2 歸一化後的嵌入
        table_embeddings = self.forward(data.x_dict, data.edge_index_dict) # [N, D]
        
        # 2. 規範化查詢向量
        query_vec_norm = F.normalize(query_vec.to(table_embeddings.device), p=2, dim=1) # [1, D]
        
        # 3. 計算點積 (餘弦相似度)
        # [1, D] @ [D, N] -> [1, N] -> [N]
        scores = torch.matmul(query_vec_norm, table_embeddings.T).squeeze(0)
        return scores


# ====================================================================
# C. 訓練流程輔助函式
# ====================================================================

def load_training_data(query_file_path: str, id_to_idx: dict):
    """
    從 JSONL 檔案載入訓練查詢，並將 table_id 轉換為 GNN 節點索引。
    
    Args:
        query_file_path (str): 'generate_query.jsonl' 檔案的路徑。
        id_to_idx (dict): 將 'table_id' (str) 映射到 GNN 節點索引 (int) 的字典。
        
    Returns:
        tuple[list[str], list[int]]: 
            - queries_text: 包含所有有效查詢問題的列表。
            - pos_indices: 每個查詢對應的正確表格 GNN 節點索引。
    """
    training_samples = []
    print(f"從 {query_file_path} 載入 queries...")
    with open(query_file_path, "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="載入 queries"):
            temp = json.loads(line)
            pos_id = temp.get('table_id')
            pos_idx = id_to_idx.get(pos_id, -1) # 查找 GNN 節點索引

            # 確保表格 ID 在我們的圖中存在
            if pos_idx != -1:
                for question in temp.get('questions', []):
                    # 確保問題不是空字串
                    if question and question.strip():
                        # 儲存 (查詢文字, 正確表格的GNN索引)
                        training_samples.append((question, pos_idx))

    if not training_samples:
        print("警告：沒有載入任何有效的訓練樣本。")
        return [], []

    # 解壓縮 (unzip) 列表
    queries_text, pos_indices = [list(t) for t in zip(*training_samples)]
    return queries_text, pos_indices

def embed_queries(queries_text: list, embedder, device):
    """
    使用 SentenceTransformer 將查詢文本列表轉換為嵌入向量。
    """
    print(f"共 {len(queries_text)} 個有效訓練樣本。開始嵌入查詢向量...")
    query_vectors_np = embedder.encode(queries_text, show_progress_bar=True)
    query_vectors = torch.tensor(query_vectors_np, dtype=torch.float, device=device)
    return query_vectors

def setup_components(embed_dim: int, metadata, hps: dict, device):
    """
    初始化模型、優化器、排程器和 GradScaler。
    """
    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps['HIDDEN_CHANNELS'],
        metadata=metadata,
        dropout=hps['DROPOUT'],
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=hps['LEARNING_RATE'], 
        weight_decay=hps['WEIGHT_DECAY']
    )
    
    # 學習率排程：先線性預熱，然後餘弦退火
    warmup_scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / float(hps['WARMUP_EPOCHS']))
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=hps['NUM_EPOCHS'] - hps['WARMUP_EPOCHS']
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[hps['WARMUP_EPOCHS']]
    )
    
    # 混合精度訓練 (AMP)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    return model, optimizer, scheduler, scaler

def compute_scores_chunked(query_emb_norm: torch.Tensor, 
                           table_emb: torch.Tensor, 
                           chunk_size: int = 1024):
    """
    分塊計算查詢和表格嵌入之間的點積，以防止 OOM。
    
    Args:
        query_emb_norm (torch.Tensor): L2 歸一化後的查詢嵌入 (shape: [Q, D])。
        table_emb (torch.Tensor): (未歸一化) 的表格嵌入 (shape: [N, D])。
                                   函數內部會處理歸一化。
        chunk_size (int): 每次處理的表格嵌入數量。
        
    Returns:
        torch.Tensor: 相似度分數矩陣 (shape: [Q, N])。
    """
    N = table_emb.size(0)
    scores_all = []
    
    # 迭代所有表格，一次處理 chunk_size 個
    for i in range(0, N, chunk_size):
        # 這裡的 chunk_emb 未經 L2 歸一化，但 model.forward() 已經歸一化了。
        # 為了保持與原始代碼功能一致，我們假設傳入的 table_emb 已經歸一化
        # (原始代碼中 model.forward() 最後有 F.normalize)
        chunk_emb = table_emb[i:i + chunk_size] # [chunk, D]
        
        # 計算查詢與當前區塊的相似度
        # [Q, D] @ [D, chunk] -> [Q, chunk]
        chunk_scores = torch.matmul(query_emb_norm, chunk_emb.T)
        scores_all.append(chunk_scores)
        
    # 將所有區塊的分數在維度 1 (表格維度) 上拼接
    return torch.cat(scores_all, dim=1) # [Q, N]


def train_one_epoch(model, data, optimizer, scaler, scheduler, 
                    query_vectors, pos_indices, hps, epoch, device):
    """
    執行一個週期的訓練。
    """
    model.train()
    total_loss = 0.0

    # 打亂訓練索引
    indices = list(range(len(query_vectors)))
    random.shuffle(indices)

    # 動態調度：計算當前 epoch 的溫度和 label smoothing
    progress = epoch / float(hps['NUM_EPOCHS'])
    curr_temp = hps['TEMP_END'] if epoch > hps['NUM_EPOCHS'] * 0.7 else hps['TEMP_START'] # 階段式衰減
    curr_smooth = hps['SMOOTH_START'] + (hps['SMOOTH_END'] - hps['SMOOTH_START']) * progress

    # 逐批訓練
    batch_iterator = tqdm(
        range(0, len(indices), hps['BATCH_SIZE']), 
        desc=f"Epoch {epoch}/{hps['NUM_EPOCHS']}"
    )
    
    for start in batch_iterator:
        end = min(start + hps['BATCH_SIZE'], len(indices))
        batch_idx = indices[start:end]
        
        q_batch = query_vectors[batch_idx]  # [B, D]
        labels = torch.tensor([pos_indices[i] for i in batch_idx], dtype=torch.long, device=device) # [B]

        optimizer.zero_grad()

        # 使用自動混合精度 (AMP)
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # [重要] 在每個 batch 重新計算 GNN 權重下的表格嵌入
            # 這是必要的，因為模型參數在更新
            table_emb = model.forward(data.x_dict, data.edge_index_dict)  # [N, D]

            # 1. 歸一化查詢批次
            q_batch_norm = F.normalize(q_batch, p=2, dim=1) # [B, D]
            
            # 2. 分塊計算 B 個查詢與 N 個表格的相似度
            logits = compute_scores_chunked(
                q_batch_norm, 
                table_emb, 
                hps['CHUNK_SIZE']
            ) # [B, N]
            
            # 3. 應用溫度
            logits /= curr_temp
            
            # 4. 計算 CrossEntropy Loss (全表格 softmax)
            loss = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)

        # 反向傳播
        scaler.scale(loss).backward()
        
        # 梯度裁剪 (在 unscale 之後, step 之前)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps['CLIP_GRAD_NORM'])
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    # 在 epoch 結束後更新學習率
    scheduler.step()
    
    avg_loss = total_loss / max(1, len(batch_iterator))
    return avg_loss, curr_temp, curr_smooth


def evaluate_retrieval(model, data, query_vectors, pos_indices, hps, device):
    """
    在所有訓練查詢上評估模型的檢索性能 (Recall@K, MRR)。
    """
    model.eval()
    all_metrics = {}
    
    with torch.no_grad():
        # 1. 計算一次 GNN 推理，獲取所有表格的最新嵌入
        table_emb_eval = model.forward(data.x_dict, data.edge_index_dict)  # [N, D]
        
        # 2. 獲取所有查詢的 L2 歸一化嵌入
        q_all_norm = F.normalize(query_vectors, p=2, dim=1)  # [Q, D]

        # 3. 分塊計算 [Q, N] 相似度矩陣
        score_mat = compute_scores_chunked(
            q_all_norm, 
            table_emb_eval, 
            hps['CHUNK_SIZE']
        ) # [Q, N]
        
        # 4. 計算指標
        topk_vals = [1, 5, 10]
        recalls = {k: 0 for k in topk_vals}
        mrr_sum = 0.0
        num_queries = score_mat.size(0)

        for q_i in range(num_queries):
            scores = score_mat[q_i]       # 當前查詢對所有表格的分數 [N]
            pos_idx = pos_indices[q_i]    # 正確答案的索引
            
            # 排序索引（由高到低）
            rank_indices = torch.argsort(scores, descending=True)
            
            # 找到正確答案的排名
            # (rank_indices == pos_idx) 產生 [False, False, ..., True, ...]
            # .nonzero() 找到 True 的位置
            rank_pos_tensor = (rank_indices == pos_idx).nonzero(as_tuple=True)[0]
            
            if rank_pos_tensor.numel() > 0:
                rank = rank_pos_tensor.item() + 1 # 排名從 1 開始
                mrr_sum += 1.0 / rank
                
                # 計算 Recall@K
                for k in topk_vals:
                    if rank <= k:
                        recalls[k] += 1
                        
        # 歸一化
        all_metrics['mrr'] = mrr_sum / num_queries
        for k in topk_vals:
            all_metrics[f'recall@{k}'] = recalls[k] / num_queries
            
    return all_metrics


# ====================================================================
# D. 訓練主迴圈 (Main Training Loop)
# ====================================================================

def main():
    # --- 1. 環境設定與超參數 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 基礎配置
    BASE_CONFIG = {
        'GRAPH_FILE': "/user_data/TabGNN/data/processed/graph.pt",
        'QUERY_FILE': "/user_data/TabGNN/data/test/feta/generate_query.jsonl",
        'MODEL_NAME': 'BAAI/bge-m3',
        'NUM_EPOCHS': 10,
        'WARMUP_EPOCHS': 2, # 縮短 warmup 以適應較短的 grid search epoch
        'BATCH_SIZE': 128,
        'CLIP_GRAD_NORM': 0.60,
        'CHUNK_SIZE': 1024,
        'TEMP_START': 0.05,
        'TEMP_END': 0.03,
        'SMOOTH_START': 0.120,
        'SMOOTH_END': 0.060,
    }

    # 定義參數網格 (Grid Search) - 擴大搜索範圍
    PARAM_GRID = {
        'LEARNING_RATE': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
        'HIDDEN_CHANNELS': [256, 512, 768],
        'DROPOUT': [0.1, 0.2, 0.3, 0.4, 0.5],
        'WEIGHT_DECAY': [1e-3, 1e-2, 5e-2]
    }

    # 生成所有參數組合
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"總共 {len(param_combinations)} 組參數組合待測試。")

    # 初始化結果 CSV
    results_file = "/user_data/TabGNN/results/grid_search_results.csv"
    with open(results_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 寫入標頭
        headers = list(keys) + ['MRR', 'Recall@1', 'Recall@5', 'Recall@10']
        writer.writerow(headers)

    # --- 2. 載入圖結構與映射表 (只載入一次) ---
    try:
        data_cpu = torch.load(BASE_CONFIG['GRAPH_FILE'], map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"錯誤：找不到 {BASE_CONFIG['GRAPH_FILE']} 檔案，請先執行 build_graph.py 建立圖。")
        return

    # 圖特徵維度
    embed_dim = data_cpu['table'].x.size(1)

    # 映射表
    try:
        id_to_idx = data_cpu.metadata_maps['table_id_to_idx']
    except (AttributeError, KeyError):
        print("錯誤：無法從 data.metadata_maps['table_id_to_idx'] 讀取映射表。")
        return

    # --- 3. 載入並嵌入訓練數據 (只載入一次) ---
    queries_text, pos_indices = load_training_data(BASE_CONFIG['QUERY_FILE'], id_to_idx)
    if not queries_text:
        return

    embedder = get_embedder(model_name=BASE_CONFIG['MODEL_NAME'], device=device)
    query_vectors = embed_queries(queries_text, embedder, device)
    del embedder
    torch.cuda.empty_cache()

    # --- 4. Grid Search 主迴圈 ---
    best_mrr = -1.0
    best_params = None
    best_model_state = None

    for idx, params in enumerate(param_combinations):
        print(f"\n=== Grid Search 組合 {idx+1}/{len(param_combinations)} ===")
        print(f"參數: {params}")
        
        # 合併基礎配置與當前參數
        current_hps = BASE_CONFIG.copy()
        current_hps.update(params)

        # 複製圖數據到 GPU (確保每次訓練都是獨立的)
        data = data_cpu.clone().to(device)

        # 初始化模型和優化器
        model, optimizer, scheduler, scaler = setup_components(
            embed_dim=embed_dim, 
            metadata=data.metadata(), 
            hps=current_hps, 
            device=device
        )
        
        # 訓練迴圈
        for epoch in range(1, current_hps['NUM_EPOCHS'] + 1):
            avg_loss, curr_temp, curr_smooth = train_one_epoch(
                model, data, optimizer, scaler, scheduler,
                query_vectors, pos_indices, current_hps, epoch, device
            )
            
            # 每個 epoch 結束後簡單打印，避免輸出過多
            if epoch % 5 == 0 or epoch == current_hps['NUM_EPOCHS']:
                 print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f}")

        # 訓練結束後評估
        print("  正在評估...")
        metrics = evaluate_retrieval(
            model, data, query_vectors, pos_indices, current_hps, device
        )
        
        current_mrr = metrics['mrr']
        print(f"  結果 -> MRR: {current_mrr:.4f} | Recall@10: {metrics['recall@10']:.4f}")

        # 寫入結果到 CSV
        with open(results_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [params[k] for k in keys] + [
                metrics['mrr'], 
                metrics['recall@1'], 
                metrics['recall@5'], 
                metrics['recall@10']
            ]
            writer.writerow(row)

        # 記錄最佳模型
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_params = params
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  [更新最佳結果] 新的 Best MRR: {best_mrr:.4f}")

        # 清理顯存
        del model, optimizer, scheduler, scaler, data
        torch.cuda.empty_cache()

    print("\n========================================")
    print("Grid Search 完成")
    print(f"最佳 MRR: {best_mrr:.4f}")
    print(f"最佳參數: {best_params}")
    print("========================================")

    # 儲存最佳模型
    if best_model_state is not None:
        save_path = "/user_data/TabGNN/checkpoints/diffusion_model_best.pt"
        save_data = {
            'model_state_dict': best_model_state,
            'hps': best_params,
            'base_config': BASE_CONFIG,
            'best_mrr': best_mrr
        }
        torch.save(save_data, save_path)
        print(f"最佳模型已儲存至 {save_path}")


if __name__ == '__main__':
    main()