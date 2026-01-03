import os
import json
import argparse
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm

# 從 retrieval.py 匯入同一份模型與嵌入器
from sentence_transformers import CrossEncoder
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
    # 移除預先計算 table_embeddings，因為 REaR 流程中會在 retrieve_with_resources 內部處理 (雖然那裡也是即時算，但為了流程一致性)
    # 不過為了 retrieve_with_resources 能夠運作，我們需要傳入 model，它內部會呼叫 model.score_tables
    # 所以這裡不需要預先計算所有 embeddings，除非我們想優化 retrieve_with_resources
    
    table_embeddings = None # 暫時不需要
    
    idx_to_id = {v: k for k, v in table_id_to_idx.items()}

    idx_to_id = {v: k for k, v in table_id_to_idx.items()}
    idx_to_id_str = {k: str(v) for k, v in idx_to_id.items()}
    mapping_keys_str: Set[str] = set(idx_to_id_str.values())

    return device, data, table_embeddings, idx_to_id_str, mapping_keys_str, model


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
            if "questions" in obj:
                question = obj.get("questions")[0].strip()
            elif "question" in obj:
                question = obj.get("question").strip()
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
    device, data, table_embeddings, idx_to_id_str, mapping_keys_str, model = load_graph_and_model(graph_path, model_path)

    # 嵌入器（同 retrieval.py）
    embedder = retrieval.get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 載入表格文字與 Reranker (REaR 需要)
    table_jsonl_path = "/user_data/TabGNN/data/table/test/feta/table.jsonl"
    table_texts = retrieval.load_table_texts(table_jsonl_path)
    
    reranker = None
    if table_texts:
        try:
            print("正在載入 Reranker (CrossEncoder)...")
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        except Exception as e:
            print(f"無法載入 Reranker: {e}")

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

    # 預先建立 col_to_table 映射以加速 retrieve_with_resources (雖然它內部也會建，但我們可以優化一下，不過為了不改動 retrieval.py 太多，我們先讓它內部建，或者我們修改 retrieval.py 接受它)
    # 這裡我們直接呼叫 retrieval.retrieve_with_resources，它會自己處理。
    # 注意：這會比原本的純向量矩陣運算慢很多，因為要跑 CrossEncoder。
    
    print("開始 REaR 評估 (這可能需要一些時間)...")
    
    # 為了避免重複 encode query，我們可以修改 retrieve_with_resources 接受 query_vec，但目前先保持簡單
    
    for i in tqdm(range(total), desc="評估中"):
        query_text = questions[i]
        relevant_ids = relevants[i]
        
        if not relevant_ids:
            continue

        # 呼叫 REaR 檢索
        # 注意：retrieve_with_resources 返回 List[Tuple[str, float]] (id, score)
        results = retrieval.retrieve_with_resources(
            query=query_text,
            top_k=top_k,
            model=model, # 這裡 model 是一個 DiffusionModel 實例
            data=data,
            embedder=embedder,
            table_texts=table_texts,
            device=device,
            idx_to_id=idx_to_id_str, # 這裡傳入的是 str key 的 dict
            reranker=reranker
        )
        
        retrieved_ids = [r[0] for r in results]
        
        hit1 += hits_at_k(retrieved_ids, relevant_ids, 1)
        hit5 += hits_at_k(retrieved_ids, relevant_ids, 5)
        hit10 += hits_at_k(retrieved_ids, relevant_ids, 10)
        mrr += reciprocal_rank(retrieved_ids, relevant_ids)

    # 彙總
    print("\n===== 評估結果 =====")
    if eval_count == 0:
        print("無可對齊的 ground truth，無法計算指標。")
        return

    results = {
        "Recall@1": hit1 / eval_count,
        "Recall@5": hit5 / eval_count,
        "Recall@10": hit10 / eval_count,
        "MRR@k": mrr / eval_count,
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
    
    evaluate(
        query_file="/user_data/TabGNN/data/table/test/feta/query.jsonl",
        model_path="/user_data/TabGNN/checkpoints/model_test.pt",
        graph_path="/user_data/TabGNN/data/processed/graph_evaluate.pt",
        top_k=10,
        batch_size=64,
        metric="cosine",
        export_csv=None,
        export_json=None,
    )


if __name__ == "__main__":
    main()
