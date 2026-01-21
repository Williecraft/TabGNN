"""評估 GNN 表格檢索模型的 Recall 和 MRR 指標"""
import json
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_model import DiffusionModel, get_embedder

# ========= 可調參數 =========
QUERY_FILE = "/user_data/TabGNN/data/table/test/feta/query.jsonl"
MODEL_PATH = "/user_data/TabGNN/checkpoints/model.pt"
GRAPH_PATH = "/user_data/TabGNN/data/processed/graph_evaluate.pt"
TOP_K = 10
# ===========================


def load_graph_and_model(graph_path: str, model_path: str):
    """載入圖結構和模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    data = torch.load(graph_path, map_location=device, weights_only=False)

    # 取得映射表
    try:
        table_id_to_idx = data.metadata_maps['table_id_to_idx']
    except Exception:
        try:
            table_ids = getattr(data['table'], 'id')
            table_id_to_idx = {tid: idx for idx, tid in enumerate(table_ids)}
        except Exception:
            table_id_to_idx = {idx: idx for idx in range(data['table'].x.size(0))}

    embed_dim = data['table'].x.size(1)

    # 載入模型和超參數
    checkpoint = torch.load(model_path, map_location=device)
    hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2, 'AGGR': 'sum'})

    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps.get('HIDDEN_CHANNELS', 128),
        metadata=data.metadata(),
        dropout=hps.get('DROPOUT', 0.2),
        sage_aggr=hps.get('SAGE_AGGR', 'sum'),
        hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    idx_to_id = {v: str(k) for k, v in table_id_to_idx.items()}

    return device, data, idx_to_id, model


def candidates_from_ground_truth(gt: Dict) -> List[str]:
    """從 ground truth 提取可能的表格 ID 候選"""
    cands = []
    # 優先使用 id 欄位
    if 'id' in gt and gt['id'] is not None:
        cands.append(str(gt['id']))
    file_name = gt.get('file_name')
    sheet_name = gt.get('sheet_name')
    if file_name and sheet_name:
        cands.append(f"{file_name}|{sheet_name}")
    if file_name:
        cands.append(str(file_name))
    if sheet_name:
        cands.append(str(sheet_name))
    seen, ordered = set(), []
    for c in cands:
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def parse_queries(query_file: str, mapping_keys: Set[str]) -> List[Tuple[str, Set[str]]]:
    """解析查詢檔案，返回 (問題, 正確答案ID集合) 列表，支援多表格 ground_truth_list"""
    queries = []
    total_lines = 0
    mappable = 0

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            obj = json.loads(line)

            if "questions" in obj:
                question = obj.get("questions")[0].strip()
            elif "question" in obj:
                question = obj.get("question").strip()
            else:
                continue

            # 從 ground_truth_list 提取所有正確表格 ID
            gt_list = obj.get("ground_truth_list", []) or []
            gt_keys: Set[str] = set()
            for gt in gt_list:
                for c in candidates_from_ground_truth(gt):
                    if c in mapping_keys:
                        gt_keys.add(c)
                        break  # 每個 gt 只取第一個匹配的候選

            if gt_keys:
                mappable += 1
            if question:
                queries.append((question, gt_keys))

    print(f"載入查詢 {total_lines} 條，其中可對齊到圖的樣本 {mappable} 條。")
    return queries


def hits_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> int:
    """檢查前 k 個檢索結果中是否有任一正確表格"""
    return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """計算 MRR，找出所有正確表格中最早出現的排名"""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


import math

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """計算 nDCG@k，支援多表格 ground truth
    
    對於每個檢索結果，如果是正確表格則 relevance=1，否則=0
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = 理想情況下所有正確表格都排在前面
    """
    # 計算 DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    
    # 計算 IDCG（理想情況：所有正確表格都排在最前面）
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """計算 Precision@k"""
    if k == 0: return 0.0
    retrieved_set = set(retrieved_ids[:k])
    intersection = retrieved_set.intersection(relevant_ids)
    return len(intersection) / k


def full_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """計算真正的 Recall@k (找回多少比例的相關文檔)"""
    if not relevant_ids: return 0.0
    retrieved_set = set(retrieved_ids[:k])
    intersection = retrieved_set.intersection(relevant_ids)
    return len(intersection) / len(relevant_ids)


def evaluate(
    query_file: str = QUERY_FILE,
    model_path: str = MODEL_PATH,
    graph_path: str = GRAPH_PATH,
    top_k: int = TOP_K
):
    """執行 GNN 評估"""
    device, data, idx_to_id, model = load_graph_and_model(graph_path, model_path)
    mapping_keys = set(idx_to_id.values())
    embedder = get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))

    queries = parse_queries(query_file, mapping_keys)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]

    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)
    
    # 初始化指標
    recall1 = recall5 = recall10 = 0.0  # Standard Recall
    exact_match5 = 0.0                  # Real Full Recall (全對才算)
    
    mrr = 0.0
    ndcg5 = ndcg10 = 0.0
    precision5 = 0.0

    print("嵌入所有查詢向量...")
    query_vecs = embedder.encode(questions, show_progress_bar=True, convert_to_tensor=True).to(device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    print("計算 GNN 表格嵌入...")
    with torch.no_grad():
        data_on_device = data.to(device)
        table_emb = model.forward(data_on_device.x_dict, data_on_device.edge_index_dict)

    print("計算相似度並評估...")
    with torch.no_grad():
        chunk_size = 256
        for start in tqdm(range(0, total, chunk_size), desc="評估中"):
            end = min(start + chunk_size, total)
            q_chunk = query_vecs[start:end]
            scores = torch.matmul(q_chunk, table_emb.T)

            for i in range(end - start):
                q_idx = start + i
                relevant_ids = relevants[q_idx]

                if not relevant_ids:
                    continue

                _, top_indices = torch.topk(scores[i], k=min(top_k, scores.size(1)))
                retrieved_ids = [idx_to_id.get(idx.item(), "") for idx in top_indices]

                # 1. Standard Recall (找回比例) - 這是學術界的 Recall
                recall1 += full_recall_at_k(retrieved_ids, relevant_ids, 1)
                recall5 += full_recall_at_k(retrieved_ids, relevant_ids, 5)
                recall10 += full_recall_at_k(retrieved_ids, relevant_ids, 10)
                
                # 2. Exact Match (真正的 Full Recall) - 必須 100% 找齊才給分
                # 檢查 Top-5 是否包含"所有"正確答案
                relevant_set = set(relevant_ids)
                retrieved_set_5 = set(retrieved_ids[:5])
                if relevant_set.issubset(retrieved_set_5):
                    exact_match5 += 1.0

                mrr += reciprocal_rank(retrieved_ids, relevant_ids)
                ndcg5 += ndcg_at_k(retrieved_ids, relevant_ids, 5)
                ndcg10 += ndcg_at_k(retrieved_ids, relevant_ids, 10)
                precision5 += precision_at_k(retrieved_ids, relevant_ids, 5)

    print("\n===== 評估結果 =====")
    if eval_count == 0:
        print("無可對齊的 ground truth，無法計算指標。")
        return

    results = {
        "Recall@1": recall1 / eval_count,
        "Recall@5": recall5 / eval_count,
        "Recall@10": recall10 / eval_count,
        "MRR@k": mrr / eval_count,
        "nDCG@5": ndcg5 / eval_count,
        "nDCG@10": ndcg10 / eval_count,
        "Precision@5": precision5 / eval_count,
        "Full Recall@5": exact_match5 / eval_count, 
    }

    for k, v in results.items():
        print(f"{k}：{v:.4f}")
    print("====================")


def main():
    evaluate()


if __name__ == "__main__":
    main()
