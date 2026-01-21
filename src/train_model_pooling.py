#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import csv
import copy
import random
import itertools

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.amp import autocast, GradScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero, GraphNorm, MessagePassing
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ========= 可調參數 =========
# 資料路徑
GRAPH_FILE = "/user_data/TabGNN/data/processed/graph_train.pt"
QUERY_FILE = "/user_data/TabGNN/data/table/train/feta/query.jsonl"
SAVE_PATH = "/user_data/TabGNN/checkpoints/model.pt"
RESULTS_FILE = "/user_data/TabGNN/results/grid_search_results.csv"

# 模型參數
MODEL_NAME = 'BAAI/bge-m3'
NUM_EPOCHS = 10
WARMUP_EPOCHS = 2
BATCH_SIZE = 128

# 訓練超參數
CLIP_GRAD_NORM = 0.60
CHUNK_SIZE = 1024
TEMP_START = 0.05
TEMP_END = 0.03
SMOOTH_START = 0.120
SMOOTH_END = 0.060

# Grid Search 設定
USE_GRID_SEARCH = False  # True: 執行 Grid Search | False: 使用最佳參數
BEST_PARAMS = {
    'LEARNING_RATE': 0.0003,
    'HIDDEN_CHANNELS': 768,
    'DROPOUT': 0.1,
    'WEIGHT_DECAY': 0.001,
    'SAGE_AGGR': 'sum',   # 鄰居聚合 (GraphSAGE 內部): 'mean', 'max', 'sum'
    'HETERO_AGGR': 'max',  # 跨邊類型聚合 (to_hetero): 'mean', 'max', 'sum'
}
# Hard Negative 設定
NUM_HARD_NEGATIVES = 3
REMINING_INTERVAL = 1  # 每幾個 epoch 重新挖掘困難負樣本
# ===========================


def get_embedder(model_name: str = MODEL_NAME, device: str = 'cuda') -> SentenceTransformer:
    """載入 SentenceTransformer 嵌入模型"""
    return SentenceTransformer(model_name, device=device)

class PoolSAGEConv(MessagePassing):
    """
    GraphSAGE-pool (paper Eq.3):
      m_u = ReLU(W_pool x_u + b)
      h_N(v) = element-wise max_{u in N(v)} m_u
      h_v' = ReLU( W · CONCAT(x_v, h_N(v)) )
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__(aggr='max')  # feature-wise max over neighbors
        self.lin_pool = nn.Linear(in_channels, out_channels, bias=True)                 # W_pool
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels, bias=True) # W
        self.dropout = dropout

    def forward(self, x, edge_index):
        # to_hetero 在 bipartite edge type 會傳 x = (x_src, x_dst)
        if isinstance(x, tuple):
            x_src, x_dst = x
            out_nei = self.propagate(edge_index, x=(x_src, x_dst))  # message 用 x_j (=src)
            self_x = x_dst                                          # update/concat 用 dst
        else:
            out_nei = self.propagate(edge_index, x=x)
            self_x = x
            
        out_nei = torch.where(torch.isfinite(out_nei), out_nei, torch.zeros_like(out_nei))

        out = torch.cat([self_x, out_nei], dim=-1)  # CONCAT(self(dst), neighbors(src-agg))
        out = self.lin_update(out)
        out = F.relu(out, inplace=True)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def message(self, x_j):
        # ★關鍵：先 affine/MLP 再做 max（符合論文 pool）
        msg = self.lin_pool(x_j)
        msg = F.relu(msg, inplace=True)
        return msg


class PoolGraphSAGE(nn.Module):
    """兩層 GraphSAGE-pool，介面與 PyG GraphSAGE 相同：forward(x, edge_index)"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2,
                 out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        assert num_layers == 2, "你目前設定 num_layers=2，這裡先用最簡單 2 層版本即可"
        if out_channels is None:
            out_channels = hidden_channels

        self.convs = nn.ModuleList([
            PoolSAGEConv(in_channels, hidden_channels, dropout=dropout),
            PoolSAGEConv(hidden_channels, out_channels, dropout=dropout),
        ])

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

class DiffusionModel(nn.Module):
    """異構圖 GNN 模型，使用 GraphSAGE 聚合節點資訊"""

    def __init__(self, embed_dim: int, hidden_channels: int, metadata, dropout: float = 0.2, 
                 sage_aggr: str = 'mean', hetero_aggr: str = 'sum'):
        super().__init__()
        self.sage = PoolGraphSAGE(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=hidden_channels,
            dropout=dropout,
        )
        self.hetero_sage = to_hetero(self.sage, metadata, aggr=hetero_aggr)
        self.norm = GraphNorm(hidden_channels)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, embed_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict):
        """前向傳播，返回 L2 歸一化的表格嵌入"""
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    def score_tables(self, data: HeteroData, query_vec: torch.Tensor):
        """計算查詢向量與所有表格的相似度分數"""
        table_embeddings = self.forward(data.x_dict, data.edge_index_dict)
        query_vec_norm = F.normalize(query_vec.to(table_embeddings.device), p=2, dim=1)
        scores = torch.matmul(query_vec_norm, table_embeddings.T).squeeze(0)
        return scores


def load_training_data(query_file_path: str, id_to_idx: dict, data: HeteroData = None, num_hard_negatives: int = 1):
    """載入訓練查詢並挖掘困難負樣本"""
    training_samples = []
    print(f"從 {query_file_path} 載入 queries...")

    # 建立鄰居索引 (用於困難負樣本挖掘)
    table_neighbors = {}
    if data is not None:
        print("正在建立鄰居索引...")
        t2c = data['table', 'has_column', 'column'].edge_index.cpu()
        c2c = data['column', 'similar_content', 'column'].edge_index.cpu()

        c2t_map = {}
        for i in range(t2c.size(1)):
            c2t_map[t2c[1, i].item()] = t2c[0, i].item()

        for i in tqdm(range(c2c.size(1)), desc="Building Graph Index"):
            c_src, c_dst = c2c[0, i].item(), c2c[1, i].item()
            if c_src in c2t_map and c_dst in c2t_map:
                t_src, t_dst = c2t_map[c_src], c2t_map[c_dst]
                if t_src != t_dst:
                    table_neighbors.setdefault(t_src, set()).add(t_dst)
        print(f"Total tables with neighbors: {len(table_neighbors)}")

    with open(query_file_path, "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="載入 queries"):
            temp = json.loads(line)

            # 取得正確表格 ID
            raw_id = None
            ground_truth_list = temp.get('ground_truth_list', [])
            if ground_truth_list and len(ground_truth_list) > 0:
                raw_id = ground_truth_list[0].get('id')
            if raw_id is None:
                raw_id = temp.get('id') or temp.get('table_id')
            if raw_id is None:
                continue

            pos_idx = id_to_idx.get(int(raw_id), -1)
            if pos_idx == -1:
                continue

            # 採樣困難負樣本
            neighbors = list(table_neighbors.get(pos_idx, []))
            if len(neighbors) >= num_hard_negatives:
                hard_negs = random.sample(neighbors, num_hard_negatives)
            else:
                hard_negs = neighbors + [-1] * (num_hard_negatives - len(neighbors))

            questions = temp.get('questions', [])
            if not questions and 'question' in temp:
                questions = [temp['question']]

            for question in questions:
                if question and question.strip():
                    training_samples.append((question, pos_idx, hard_negs))

    if not training_samples:
        print("警告：沒有載入任何有效的訓練樣本。")
        return [], [], []

    queries_text, pos_indices, hard_neg_indices = [list(t) for t in zip(*training_samples)]

    # 驗證索引範圍
    if data is not None:
        num_tables = data['table'].num_nodes
        valid_samples = [(q, p, [idx if idx < num_tables else -1 for idx in h])
                         for q, p, h in zip(queries_text, pos_indices, hard_neg_indices)
                         if p < num_tables]
        if len(valid_samples) < len(queries_text):
            queries_text, pos_indices, hard_neg_indices = [list(t) for t in zip(*valid_samples)]
            print(f"Filtered to {len(queries_text)} valid samples.")

    return queries_text, pos_indices, hard_neg_indices


def mine_hard_negatives_topk(model, data, query_vectors, pos_indices, num_hard_negatives=5, device='cuda'):
    """Query-Aware 困難負樣本挖掘：找出每個查詢最容易搞混的 K 張錯誤表格"""
    model.eval()
    with torch.no_grad():
        table_emb = F.normalize(model.forward(data.x_dict, data.edge_index_dict), p=2, dim=1)
        q_norm = F.normalize(query_vectors, p=2, dim=1)

        num_tables = table_emb.size(0)
        actual_k = min(num_hard_negatives, num_tables - 1)

        all_topk_indices = []
        chunk_size = 1024

        for start in range(0, len(query_vectors), chunk_size):
            end = min(start + chunk_size, len(query_vectors))
            q_chunk = q_norm[start:end]
            sim_chunk = torch.matmul(q_chunk, table_emb.T)

            for i in range(end - start):
                pos_idx = pos_indices[start + i]
                if 0 <= pos_idx < num_tables:
                    sim_chunk[i, pos_idx] = -float('inf')

            _, topk_idx = torch.topk(sim_chunk, k=actual_k, dim=1)
            all_topk_indices.extend(topk_idx.tolist())

        if actual_k < num_hard_negatives:
            padding = [-1] * (num_hard_negatives - actual_k)
            all_topk_indices = [negs + padding for negs in all_topk_indices]

    model.train()
    return all_topk_indices


def embed_queries(queries_text: list, embedder, device):
    """嵌入查詢文本"""
    print(f"共 {len(queries_text)} 個訓練樣本。開始嵌入查詢向量...")
    query_vectors_np = embedder.encode(queries_text, show_progress_bar=True)
    return torch.tensor(query_vectors_np, dtype=torch.float, device=device)


def setup_components(embed_dim: int, metadata, hps: dict, device):
    """初始化模型、優化器和排程器"""
    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps['HIDDEN_CHANNELS'],
        metadata=metadata,
        dropout=hps['DROPOUT'],
        sage_aggr=hps.get('SAGE_AGGR', 'mean'),
        hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hps['LEARNING_RATE'], weight_decay=hps['WEIGHT_DECAY'])

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(hps['WARMUP_EPOCHS'])))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=hps['NUM_EPOCHS'] - hps['WARMUP_EPOCHS'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[hps['WARMUP_EPOCHS']])

    scaler = GradScaler(enabled=(device.type == 'cuda'))
    return model, optimizer, scheduler, scaler


def compute_scores_chunked(query_emb_norm: torch.Tensor, table_emb: torch.Tensor, chunk_size: int = 1024):
    """分塊計算相似度矩陣"""
    scores_all = []
    for i in range(0, table_emb.size(0), chunk_size):
        chunk_scores = torch.matmul(query_emb_norm, table_emb[i:i + chunk_size].T)
        scores_all.append(chunk_scores)
    return torch.cat(scores_all, dim=1)


def train_one_epoch(model, data, optimizer, scaler, scheduler, query_vectors, pos_indices, hard_neg_indices, hps, epoch, device):
    """執行一個訓練週期"""
    model.train()
    total_loss = 0.0
    indices = list(range(len(query_vectors)))
    random.shuffle(indices)

    progress = epoch / float(hps['NUM_EPOCHS'])
    curr_temp = hps['TEMP_END'] if epoch > hps['NUM_EPOCHS'] * 0.7 else hps['TEMP_START']
    curr_smooth = hps['SMOOTH_START'] + (hps['SMOOTH_END'] - hps['SMOOTH_START']) * progress

    batch_iterator = tqdm(range(0, len(indices), hps['BATCH_SIZE']), desc=f"Epoch {epoch}/{hps['NUM_EPOCHS']}")

    for start in batch_iterator:
        end = min(start + hps['BATCH_SIZE'], len(indices))
        batch_idx = indices[start:end]
        q_batch = query_vectors[batch_idx]
        labels = torch.tensor([pos_indices[i] for i in batch_idx], dtype=torch.long, device=device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            table_emb = model.forward(data.x_dict, data.edge_index_dict)
            q_batch_norm = F.normalize(q_batch, p=2, dim=1)
            logits = compute_scores_chunked(q_batch_norm, table_emb, hps['CHUNK_SIZE']) / curr_temp

            loss_in_batch = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)

            # Hard Negative Margin Loss
            loss_hard = 0.0
            if hps.get('USE_HARD_NEG', False):
                batch_hard_negs = [hard_neg_indices[i] for i in batch_idx]
                hard_negs_tensor = torch.tensor(batch_hard_negs, device=device, dtype=torch.long)
                mask = (hard_negs_tensor != -1)
                safe_hard_negs = hard_negs_tensor.clone()
                safe_hard_negs[~mask] = 0

                pos_scores = logits[range(len(batch_idx)), labels]
                neg_scores = torch.gather(logits, 1, safe_hard_negs)

                margin = 0.2 / curr_temp
                losses = F.relu(neg_scores - pos_scores.unsqueeze(1) + margin) * mask.float()
                loss_hard = losses.sum() / (mask.sum() + 1e-9)

            loss = loss_in_batch + 0.5 * loss_hard

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps['CLIP_GRAD_NORM'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / max(1, len(batch_iterator)), curr_temp, curr_smooth


def evaluate_retrieval(model, data, query_vectors, pos_indices, hps, device):
    """評估模型的檢索性能"""
    model.eval()
    with torch.no_grad():
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_all_norm = F.normalize(query_vectors, p=2, dim=1)
        score_mat = compute_scores_chunked(q_all_norm, table_emb, hps['CHUNK_SIZE'])

        topk_vals = [1, 5, 10]
        recalls = {k: 0 for k in topk_vals}
        mrr_sum = 0.0

        for q_i in range(score_mat.size(0)):
            rank_indices = torch.argsort(score_mat[q_i], descending=True)
            rank_pos = (rank_indices == pos_indices[q_i]).nonzero(as_tuple=True)[0]

            if rank_pos.numel() > 0:
                rank = rank_pos.item() + 1
                mrr_sum += 1.0 / rank
                for k in topk_vals:
                    if rank <= k:
                        recalls[k] += 1

    num_queries = score_mat.size(0)
    return {
        'mrr': mrr_sum / num_queries,
        **{f'recall@{k}': recalls[k] / num_queries for k in topk_vals}
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 建立參數組合
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
        param_combinations = [BEST_PARAMS]
        keys = list(BEST_PARAMS.keys())

    print(f"總共 {len(param_combinations)} 組參數組合。")

    # 初始化結果 CSV
    with open(RESULTS_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(list(keys) + ['MRR', 'Recall@1', 'Recall@5', 'Recall@10'])

    # 載入圖
    try:
        data_cpu = torch.load(GRAPH_FILE, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"錯誤：找不到 {GRAPH_FILE}，請先執行 build_graph.py。")
        return

    embed_dim = data_cpu['table'].x.size(1)

    try:
        id_to_idx = data_cpu.metadata_maps['table_id_to_idx']
    except (AttributeError, KeyError):
        print("錯誤：無法讀取映射表。")
        return

    # 載入訓練數據
    queries_text, pos_indices, hard_neg_indices = load_training_data(QUERY_FILE, id_to_idx, data=data_cpu, num_hard_negatives=NUM_HARD_NEGATIVES)
    if not queries_text:
        return

    embedder = get_embedder(model_name=MODEL_NAME, device=device)
    query_vectors = embed_queries(queries_text, embedder, device)
    del embedder
    torch.cuda.empty_cache()

    # 訓練迴圈
    best_mrr = -1.0
    best_params = None
    best_model_state = None

    base_config = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'CHUNK_SIZE': CHUNK_SIZE,
        'TEMP_START': TEMP_START,
        'TEMP_END': TEMP_END,
        'SMOOTH_START': SMOOTH_START,
        'SMOOTH_END': SMOOTH_END,
    }

    for idx, params in enumerate(param_combinations):
        print(f"\n=== {idx + 1}/{len(param_combinations)} ===")
        print(f"參數: {params}")

        current_hps = {**base_config, **params, 'USE_HARD_NEG': True}
        data = data_cpu.clone().to(device)
        model, optimizer, scheduler, scaler = setup_components(embed_dim, data.metadata(), current_hps, device)
        current_hard_neg_indices = hard_neg_indices.copy()

        for epoch in range(1, current_hps['NUM_EPOCHS'] + 1):
            if epoch > 1 and epoch % REMINING_INTERVAL == 0:
                current_hard_neg_indices = mine_hard_negatives_topk(
                    model, data, query_vectors, pos_indices,
                    num_hard_negatives=NUM_HARD_NEGATIVES, device=device
                )

            avg_loss, _, _ = train_one_epoch(
                model, data, optimizer, scaler, scheduler,
                query_vectors, pos_indices, current_hard_neg_indices, current_hps, epoch, device
            )

            if epoch % 5 == 0 or epoch == current_hps['NUM_EPOCHS']:
                print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f}")

        metrics = evaluate_retrieval(model, data, query_vectors, pos_indices, current_hps, device)
        print(f"  結果 -> MRR: {metrics['mrr']:.4f} | Recall@10: {metrics['recall@10']:.4f}")

        with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([params[k] for k in keys] + [metrics['mrr'], metrics['recall@1'], metrics['recall@5'], metrics['recall@10']])

        if metrics['mrr'] > best_mrr:
            best_mrr = metrics['mrr']
            best_params = params
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  [更新最佳結果] Best MRR: {best_mrr:.4f}")

        del model, optimizer, scheduler, scaler, data
        torch.cuda.empty_cache()

    print(f"\n========================================")
    print(f"完成！最佳 MRR: {best_mrr:.4f}")
    print(f"最佳參數: {best_params}")
    print("========================================")

    if best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'hps': best_params,
            'best_mrr': best_mrr
        }, SAVE_PATH)
        print(f"模型已儲存至 {SAVE_PATH}")


if __name__ == '__main__':
    main()