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

def load_training_data(query_file_path: str, id_to_idx: dict, data: HeteroData = None, num_hard_negatives: int = 1):
    """
    從 JSONL 檔案載入訓練查詢，並將 table_id 轉換為 GNN 節點索引。
    同時利用圖結構挖掘困難負樣本 (Hard Negatives)。
    
    Args:
        query_file_path (str): 'generate_query.jsonl' 檔案的路徑。
        id_to_idx (dict): 將 'table_id' (str) 映射到 GNN 節點索引 (int) 的字典。
        data (HeteroData, optional): 圖數據，用於查找鄰居作為困難負樣本。
        num_hard_negatives (int): 每個正樣本採樣多少個困難負樣本。
        
    Returns:
        tuple: 
            - queries_text: 查詢文字列表。
            - pos_indices: 正確表格索引列表。
            - hard_neg_indices: 困難負樣本索引列表 (List[List[int]])。
    """
    training_samples = []
    print(f"從 {query_file_path} 載入 queries (含 Hard Negatives)...")
    
    # 預先處理圖的鄰居關係，加速查找
    # 建立 Table -> Neighbors (via similar_content) 的映射
    table_neighbors = {}
    if data is not None:
        print("正在建立鄰居索引以進行困難負樣本挖掘...")
        # 1. Table -> Column
        t2c = data['table', 'has_column', 'column'].edge_index.cpu()
        # 2. Column -> Column (similar)
        c2c = data['column', 'similar_content', 'column'].edge_index.cpu()
        # 3. Column -> Table (reverse)
        # 建立 col_idx -> table_idx 映射
        c2t_map = {}
        for i in range(t2c.size(1)):
            t = t2c[0, i].item()
            c = t2c[1, i].item()
            c2t_map[c] = t
            
        # 建立 table_idx -> set(neighbor_table_idxs)
        # 這一步可能比較慢，但只跑一次
        # 優化：直接遍歷 c2c
        for i in tqdm(range(c2c.size(1)), desc="Building Graph Index"):
            c_src = c2c[0, i].item()
            c_dst = c2c[1, i].item()
            
            if c_src in c2t_map and c_dst in c2t_map:
                t_src = c2t_map[c_src]
                t_dst = c2t_map[c_dst]
                
                if t_src != t_dst:
                    if t_src not in table_neighbors:
                        table_neighbors[t_src] = set()
                    table_neighbors[t_src].add(t_dst)
        print(f"Total tables with neighbors: {len(table_neighbors)}")

    with open(query_file_path, "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="載入 queries"):
            temp = json.loads(line)
            
            # 取得正確表格 ID：優先從 ground_truth_list 取得，否則從 id/table_id 取得
            raw_id = None
            ground_truth_list = temp.get('ground_truth_list', [])
            if ground_truth_list and isinstance(ground_truth_list, list) and len(ground_truth_list) > 0:
                # 從 ground_truth_list 的第一個元素取得 id
                raw_id = ground_truth_list[0].get('id')
            
            # 如果 ground_truth_list 沒有 id，則嘗試 temp 的 id 或 table_id
            if raw_id is None:
                raw_id = temp.get('id') or temp.get('table_id')
            
            if raw_id is None:
                continue
            
            pos_id = int(raw_id)
            pos_idx = id_to_idx.get(pos_id, -1)

            if pos_idx != -1:
                # 採樣困難負樣本
                hard_negs = []
                if pos_idx in table_neighbors:
                    neighbors = list(table_neighbors[pos_idx])
                    if len(neighbors) >= num_hard_negatives:
                        hard_negs = random.sample(neighbors, num_hard_negatives)
                    else:
                        # 如果鄰居不夠，重複採樣或補隨機 (這裡簡單處理：有幾個算幾個，不夠的補 -1)
                        hard_negs = neighbors + [-1] * (num_hard_negatives - len(neighbors))
                else:
                    # print(f"Table {pos_id} (idx={pos_idx}) has no neighbors.")
                    hard_negs = [-1] * num_hard_negatives # 無鄰居

                questions = temp.get('questions', [])
                if not questions and 'question' in temp:
                    questions = [temp['question']]
                
                for question in questions:
                    if question and question.strip():
                        training_samples.append((question, pos_idx, hard_negs))
            else:
                 # print(f"Table {pos_id} not found in id_to_idx.")
                 pass

    if not training_samples:
        print("警告：沒有載入任何有效的訓練樣本。")
        return [], [], []

    queries_text, pos_indices, hard_neg_indices = [list(t) for t in zip(*training_samples)]
    
    # Safety check
    if data is not None:
        num_tables = data['table'].num_nodes
        max_idx = max(pos_indices)
        if max_idx >= num_tables:
            print(f"Error: Max pos_idx {max_idx} exceeds num_tables {num_tables}")
            # Filter out invalid samples
            valid_samples = []
            for q, p, h in zip(queries_text, pos_indices, hard_neg_indices):
                if p < num_tables:
                    valid_h = [idx for idx in h if idx < num_tables]
                    # Pad if needed
                    if len(valid_h) < num_hard_negatives:
                         valid_h += [-1] * (num_hard_negatives - len(valid_h))
                    valid_samples.append((q, p, valid_h))
            
            if not valid_samples:
                return [], [], []
            queries_text, pos_indices, hard_neg_indices = [list(t) for t in zip(*valid_samples)]
            print(f"Filtered to {len(queries_text)} valid samples.")

    return queries_text, pos_indices, hard_neg_indices


def mine_hard_negatives_topk(model, data, query_vectors, pos_indices, 
                              num_hard_negatives=5, device='cuda'):
    """
    Query-Aware 困難負樣本挖掘：
    對每個查詢，用模型當前的向量找出最相似的 K 張錯誤表格。
    
    這比使用圖鄰居更精準，因為它直接針對每個 Query 找出模型最容易搞混的表格。
    
    Args:
        model: 訓練中的 GNN 模型
        data: PyG HeteroData 圖資料
        query_vectors: 查詢向量 [Q, D]
        pos_indices: 每個查詢對應的正確表格索引
        num_hard_negatives: 要挖掘的困難負樣本數量
        device: 運算裝置
        
    Returns:
        List[List[int]]: 每個查詢的 K 個困難負樣本索引
    """
    model.eval()
    with torch.no_grad():
        # 計算所有表格的向量
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        table_emb = F.normalize(table_emb, p=2, dim=1)  # [N, D]
        q_norm = F.normalize(query_vectors, p=2, dim=1)  # [Q, D]
        
        num_tables = table_emb.size(0)
        num_queries = q_norm.size(0)
        
        # 限制 num_hard_negatives 不超過表格數量減一
        actual_k = min(num_hard_negatives, num_tables - 1)
        
        # 分塊計算相似度以節省記憶體
        chunk_size = 1024
        all_topk_indices = []
        
        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            q_chunk = q_norm[start:end]  # [chunk, D]
            
            # 計算相似度矩陣: [chunk, N]
            sim_chunk = torch.matmul(q_chunk, table_emb.T)
            
            # 把正確答案的分數設為負無窮大（排除它）
            for i in range(end - start):
                pos_idx = pos_indices[start + i]
                if 0 <= pos_idx < num_tables:
                    sim_chunk[i, pos_idx] = -float('inf')
            
            # 取出每個查詢的 Top-K
            _, topk_idx = torch.topk(sim_chunk, k=actual_k, dim=1)
            all_topk_indices.extend(topk_idx.tolist())
        
        # 如果 actual_k 小於要求的數量，補 -1
        if actual_k < num_hard_negatives:
            padding = [-1] * (num_hard_negatives - actual_k)
            all_topk_indices = [negs + padding for negs in all_topk_indices]
        
    model.train()
    return all_topk_indices


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
                    query_vectors, pos_indices, hard_neg_indices, hps, epoch, device):
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
            # 這是原本的 In-batch Negatives Loss
            loss_in_batch = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)
            
            # 5. 加入 Hard Negative Loss
            # 我們希望 Query 與 Hard Negative 的相似度越低越好
            # 這裡使用一個簡單的 Margin Ranking Loss 概念，或者直接將 Hard Negatives 視為額外的負樣本類別
            # 為了簡單且有效，我們這裡採用 "InfoNCE with Hard Negatives" 的變體
            # 即：最大化 (Q . Pos) / (Sum(Q . Negs))
            # 但由於 logits 已經是全表格的相似度，其實 Hard Negatives 已經在分母裡了 (因為它們也是表格)
            # 只是 In-batch Negatives 只有 Batch Size 個，而全表格 Softmax 包含了所有表格
            # 等等，原本的 logits 是 [B, N]，N 是所有表格。
            # 所以 F.cross_entropy(logits, labels) 其實已經包含了所有表格作為負樣本！
            # 這意味著 Hard Negatives (作為 N 中的一部分) 已經被考慮進去了。
            # 
            # 那麼，為什麼還需要特別挖掘 Hard Negatives？
            # 因為在標準的 InfoNCE 中，通常只用 In-batch negatives (分母只有 B 個)。
            # 但這裡我們用了 `compute_scores_chunked` 算出了 [B, N] 的所有分數。
            # 所以我們的 Loss 其實已經是 "Global Softmax" 了！這比 In-batch negatives 強很多。
            # 
            # 不過，為了進一步強化，我們可以對 Hard Negatives 加重懲罰 (Reweighting)。
            # 或者，如果我們改用 Contrastive Loss (Margin Loss)，Hard Negatives 就很有用。
            # 
            # 讓我們採用 "Hard Negative Reweighting" 策略：
            # 在計算 CrossEntropy 時，雖然所有負樣本都在，但模型可能對 Hard Negatives 的梯度不夠大。
            # 我們可以額外加一個 Margin Loss： max(0, sim(Q, Neg) - sim(Q, Pos) + margin)
            
            loss_hard = 0.0
            if hps.get('USE_HARD_NEG', False):
                # 取出對應的 Hard Negatives 索引
                # hard_neg_indices 是 List[List[int]]，長度為 len(query_vectors)
                # 我們需要取出當前 batch 的 hard negs
                batch_hard_negs = [hard_neg_indices[i] for i in batch_idx] # [B, K]
                
                # 扁平化處理，並過濾掉 -1
                # 為了向量化計算，我們需要從 logits 中取出這些位置的分數
                # logits: [B, N]
                
                # 建立一個 mask 或 gather 索引
                # 這裡為了簡單，我們逐個樣本計算 Margin Loss
                margin = 0.2
                
                # 取得正樣本分數: logits[b, label]
                # 取得負樣本分數: logits[b, hard_neg]
                
                # 由於 logits 已經除以 temp，我們還原一下或者直接用
                # 通常 Margin Loss 作用在原始 cosine similarity 上比較好控制 margin
                # 但這裡為了方便，直接用 logits (scaled similarity)
                
                # 重新計算原始相似度 (不除 temp) 比較安全
                # 但為了效能，我們直接用 logits * temp
                
                pos_scores = logits[range(len(batch_idx)), labels] # [B]
                
                # 收集負樣本分數
                # 這裡稍微麻煩一點，因為每個樣本的負樣本數量可能不同 (雖然我們補了 -1)
                # 且我們需要從 logits 裡 gather
                
                # 轉換 batch_hard_negs 為 tensor
                # 注意：-1 的索引會報錯，所以要處理
                # 先轉成 tensor，把 -1 換成 0 (或其他無害索引)，然後 mask 掉 loss
                
                hard_negs_tensor = torch.tensor(batch_hard_negs, device=device, dtype=torch.long) # [B, K]
                mask = (hard_negs_tensor != -1)
                safe_hard_negs = hard_negs_tensor.clone()
                safe_hard_negs[~mask] = 0 # 避免 gather 越界
                
                # gather 負樣本分數: [B, K]
                neg_scores = torch.gather(logits, 1, safe_hard_negs)
                
                # Margin Loss: max(0, neg - pos + margin/temp)
                # 因為 logits 是 sim/temp，所以 margin 也要除以 temp
                target_margin = margin / curr_temp
                
                losses = F.relu(neg_scores - pos_scores.unsqueeze(1) + target_margin)
                
                # 只計算有效的負樣本
                losses = losses * mask.float()
                
                loss_hard = losses.sum() / (mask.sum() + 1e-9)
            
            loss = loss_in_batch + 0.5 * loss_hard # 0.5 是權重，可調

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

    # ========================================
    # 訓練模式選擇
    # ========================================
    USE_GRID_SEARCH = False  # True: 執行 Grid Search | False: 使用最佳參數直接訓練
    
    # 最佳參數 (Grid Search 結果)
    BEST_PARAMS = {
        'LEARNING_RATE': 0.0003,
        'HIDDEN_CHANNELS': 768,
        'DROPOUT': 0.1,
        'WEIGHT_DECAY': 0.001
    }

    # 配置
    BASE_CONFIG = {
        'GRAPH_FILE': "/user_data/TabGNN/data/processed/graph_evaluate.pt",
        'QUERY_FILE': "/user_data/TabGNN/data/table/test/feta/query.jsonl",
        'MODEL_NAME': 'BAAI/bge-m3',
        'NUM_EPOCHS': 10,
        'WARMUP_EPOCHS': 2,
        'BATCH_SIZE': 128,
        'CLIP_GRAD_NORM': 0.60,
        'CHUNK_SIZE': 1024,
        'TEMP_START': 0.05,
        'TEMP_END': 0.03,
        'SMOOTH_START': 0.120,
        'SMOOTH_END': 0.060,
    }

    # Grid Search 
    if USE_GRID_SEARCH:
        PARAM_GRID = {
            'LEARNING_RATE': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            'HIDDEN_CHANNELS': [256, 512, 768],
            'DROPOUT': [0.1, 0.2, 0.3, 0.4, 0.5],
            'WEIGHT_DECAY': [1e-3, 1e-2, 5e-2]
        }
        keys, values = zip(*PARAM_GRID.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        # 直接使用最佳參數
        param_combinations = [BEST_PARAMS]
        keys = list(BEST_PARAMS.keys())
    
    print(f"總共 {len(param_combinations)} 組參數組合待測試。")

    # 初始化結果 CSV
    results_file = "/user_data/TabGNN/results/grid_search_results.csv"
    with open(results_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 寫入標頭
        headers = list(keys) + ['MRR', 'Recall@1', 'Recall@5', 'Recall@10']
        writer.writerow(headers)

    # --- 2. 載入圖與映射表 ---
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
    # 傳入 data 以便挖掘 Hard Negatives
    queries_text, pos_indices, hard_neg_indices = load_training_data(BASE_CONFIG['QUERY_FILE'], id_to_idx, data=data_cpu, num_hard_negatives=3)
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
        print(f"\n=== {idx+1}/{len(param_combinations)} ===")
        print(f"參數: {params}")
        
        # 合併基礎配置與當前參數
        current_hps = BASE_CONFIG.copy()
        current_hps.update(params)
        current_hps['USE_HARD_NEG'] = True # 啟用 Hard Negative Loss
        
        # Query-Aware Hard Negative 挖掘間隔 (每幾個 epoch 重新挖掘)
        REMINING_INTERVAL = 1

        # 複製圖數據到 GPU (確保每次訓練都是獨立的)
        data = data_cpu.clone().to(device)

        # 初始化模型和優化器
        model, optimizer, scheduler, scaler = setup_components(
            embed_dim=embed_dim, 
            metadata=data.metadata(), 
            hps=current_hps, 
            device=device
        )
        
        # 複製初始困難負樣本索引（這樣不同 grid search 組合之間互不影響）
        current_hard_neg_indices = hard_neg_indices.copy()
        
        # 訓練迴圈
        for epoch in range(1, current_hps['NUM_EPOCHS'] + 1):
            # 定期使用 Query-Aware 方法重新挖掘困難負樣本
            if epoch > 1 and epoch % REMINING_INTERVAL == 0:
                current_hard_neg_indices = mine_hard_negatives_topk(
                    model, data, query_vectors, pos_indices,
                    num_hard_negatives=3, device=device
                )
            
            avg_loss, curr_temp, curr_smooth = train_one_epoch(
                model, data, optimizer, scaler, scheduler,
                query_vectors, pos_indices, current_hard_neg_indices, current_hps, epoch, device
            )
            
            # 每個 epoch 結束後簡單打印，避免輸出過多
            if epoch % 5 == 0 or epoch == current_hps['NUM_EPOCHS']:
                 print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f}")

        # 訓練結束後評估
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
        save_path = "/user_data/TabGNN/checkpoints/model_test.pt"
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