import os
import json
import argparse
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm

# 從 retrieval.py 匯入同一份模型與嵌入器
import retrieval


def load_graph_and_model(graph_path: str, model_path: str, use_cache: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")
    try:
        data = torch.load(graph_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise

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
    try:
        checkpoint = torch.load(model_path, map_location=device)
        hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2}) # 向下相容
        hidden_channels = hps.get('HIDDEN_CHANNELS', 128)
        dropout = hps.get('DROPOUT', 0.2)
    except FileNotFoundError:
        raise

    model = retrieval.DiffusionModel(
        embed_dim=embed_dim, 
        hidden_channels=hidden_channels, 
        metadata=data.metadata(),
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        table_embeddings = model.forward(data.x_dict, data.edge_index_dict)

    idx_to_id = {v: k for k, v in table_id_to_idx.items()}
    idx_to_id_str = {k: str(v) for k, v in idx_to_id.items()}
    mapping_keys_str: Set[str] = set(idx_to_id_str.values())

    return device, data, table_embeddings, idx_to_id_str, mapping_keys_str


def candidates_from_ground_truth(gt: Dict) -> List[str]:
    cands = []
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
    # 去重並保留順序
    seen, ordered = set(), []
    for c in cands:
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def parse_queries(query_file: str, mapping_keys_str: Set[str]) -> List[Tuple[str, Set[str]]]:
    queries: List[Tuple[str, Set[str]]] = []
    total_lines = 0
    mappable = 0

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            obj = json.loads(line)
            question = obj.get("question", "").strip()
            gt_list = obj.get("ground_truth_list", []) or []

            # 建立 ground truth 可對齊到圖的鍵集合
            gt_keys: Set[str] = set()
            for gt in gt_list:
                for c in candidates_from_ground_truth(gt):
                    if c in mapping_keys_str:
                        gt_keys.add(c)
                        break  # 對每個 gt 只取第一個可對齊鍵

            if gt_keys:
                mappable += 1
            else:
                # 若完全對不上，仍加入空集合，後續計算時記為 miss
                pass

            if question:
                queries.append((question, gt_keys))

    print(f"載入查詢 {total_lines} 條，其中可對齊到圖的樣本 {mappable} 條。")
    return queries


def hits_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> int:
    topk = retrieved_ids[:k]
    return int(any(rid in relevant_ids for rid in topk))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_ids:
            hits += 1
            sum_prec += hits / i
    # 若 relevant 超過 k，這是 AP@k（截斷）
    return sum_prec / min(len(relevant_ids), k)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    def dcg(scores: List[int]) -> float:
        s = 0.0
        for i, rel in enumerate(scores):
            denom = torch.log2(torch.tensor(i + 2.0)).item()
            s += (rel / denom)
        return s

    rels = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    dcg_val = dcg(rels)
    ideal_rels = sorted(rels, reverse=True)  # 二元相關，理想是全 1 在最前
    idcg_val = dcg(ideal_rels)
    return (dcg_val / idcg_val) if idcg_val > 0 else 0.0


def evaluate(query_file: str, model_path: str, graph_path: str, top_k: int = 10, batch_size: int = 64, metric: str = "cosine", export_csv: str = None, export_json: str = None, out_dir: str = None):
    device, data, table_embeddings, idx_to_id_str, mapping_keys_str = load_graph_and_model(graph_path, model_path)

    # 嵌入器（同 retrieval.py）
    embedder = retrieval.get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))

    # 解析查詢
    queries = parse_queries(query_file, mapping_keys_str)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]

    # 指標累計
    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)  # 只統計可對齊樣本
    hit1 = hit5 = hit10 = 0
    mrr = 0.0
    map_k = 0.0
    ndcg10 = 0.0

    # 批次嵌入查詢以加速
    print("開始嵌入查詢向量...")
    query_embs = embedder.encode(questions, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
    query_embs = query_embs.to(device)
    query_embs = F.normalize(query_embs, p=2, dim=1)

    with torch.no_grad():
        for i in tqdm(range(total), desc="評估中"):
            q_vec = query_embs[i].unsqueeze(0)
            if metric == "l2":
                scores = -torch.cdist(q_vec, table_embeddings, p=2).squeeze(0)
            else:
                scores = torch.matmul(q_vec, table_embeddings.T).squeeze(0)
            k_val = min(top_k, scores.numel())
            top_scores, top_indices = torch.topk(scores, k=k_val, largest=True)

            # 轉成 table_id 字串
            retrieved_ids = [idx_to_id_str[idx.item()] for idx in top_indices]
            relevant_ids = relevants[i]

            # 只有在 relevant 可對齊時才計入指標
            if relevant_ids:
                hit1 += hits_at_k(retrieved_ids, relevant_ids, 1)
                hit5 += hits_at_k(retrieved_ids, relevant_ids, 5)
                hit10 += hits_at_k(retrieved_ids, relevant_ids, 10)
                mrr += reciprocal_rank(retrieved_ids, relevant_ids)
                map_k += average_precision_at_k(retrieved_ids, relevant_ids, top_k)
                ndcg10 += ndcg_at_k(retrieved_ids, relevant_ids, 10)

    # 彙總
    print("\n===== 評估結果 =====")
    print(f"總查詢數：{total}")
    print(f"可對齊評估的查詢數：{eval_count}")
    if eval_count == 0:
        print("無可對齊的 ground truth，無法計算指標。")
        return

    results = {
        "total": total,
        "eval_count": eval_count,
        "Recall@1": hit1 / eval_count,
        "Recall@5": hit5 / eval_count,
        "Recall@10": hit10 / eval_count,
        "MRR@k": mrr / eval_count,
        "MAP@k": map_k / eval_count,
        "NDCG@10": ndcg10 / eval_count,
    }

    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}：{v:.4f}")
        else:
            print(f"{k}：{v}")
    print("====================")

    if export_csv:
        import csv
        with open(export_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in results.items():
                writer.writerow([k, v])
    if export_json:
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="批次評估檢索指標（Recall/MRR/MAP/NDCG/導出/度量選項）")
    parser.add_argument("--query_file", type=str, default="/user_data/TabGNN/data/table/test/feta/query.jsonl")
    parser.add_argument("--model_path", type=str, default="/user_data/TabGNN/checkpoints/diffusion_model_best.pt")
    parser.add_argument("--graph_path", type=str, default="/user_data/TabGNN/data/processed/graph.pt")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "dot", "l2"])
    parser.add_argument("--export_csv", type=str, default=None)
    parser.add_argument("--export_json", type=str, default=None)
    args = parser.parse_args()

    evaluate(
        query_file=args.query_file,
        model_path=args.model_path,
        graph_path=args.graph_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        metric=args.metric,
        export_csv=args.export_csv,
        export_json=args.export_json,
    )


if __name__ == "__main__":
    main()
