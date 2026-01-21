import json
import numpy as np
from pathlib import Path
from typing import List, Dict

# ========= 可調參數 =========
SOURCE = "train/spider_multitabqa"
FILENAME = "query_ollama.npz"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
DEVICE = "cuda"   # "cuda" / "cuda:0" / None
TOP_KS = (1, 5, 10)

VEC_PATH = Path(f"/user_data/TabGNN/data/embeddings/{SOURCE}/{FILENAME}")
GOLD_PATH = Path(f"/user_data/TabGNN/data/table/{SOURCE}/query.jsonl")

SKIP_OOV = True  # ground truth table 不在向量庫時：是否從正解集合移除；若移除後為空則跳過該 query
# ===========================


def embed_texts(model_name: str, texts: list, batch_size: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine = dot
    )
    return embs.astype(np.float32)


def load_gold_queries(path: Path) -> List[Dict]:
    """
    讀 query.jsonl
    每筆輸出：
      {
        "qid": int,          # query 本身 id（可有可無，僅方便追蹤）
        "question": str,
        "gt_ids": List[int], # 以 ground_truth_list 的 id 當作正解集合（可多個）
      }
    """
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            if not q:
                continue

            gt_list = obj.get("ground_truth_list") or []
            gt_ids = []
            for g in gt_list:
                if isinstance(g, dict) and ("id" in g):
                    try:
                        gt_ids.append(int(g["id"]))
                    except Exception:
                        pass

            # 沒有 ground_truth_list 就跳過（或你也可以改成 fallback 用 obj["id"]）
            if not gt_ids:
                continue

            qid = obj.get("id")
            qid = int(qid) if qid is not None else -1

            items.append({"qid": qid, "question": q, "gt_ids": gt_ids})
    return items


def average_precision(sorted_rels: np.ndarray, R: int) -> float:
    """
    sorted_rels: 長度 N 的 0/1，代表排序後每個位置是否 relevant
    R: relevant 總數（GT size）
    """
    if R <= 0:
        return 0.0
    # AP = (1/R) * sum_{i where rel_i=1} Precision@i
    rel_pos = np.flatnonzero(sorted_rels)
    if rel_pos.size == 0:
        return 0.0
    precisions = []
    hit = 0
    for idx in rel_pos:
        hit += 1
        precisions.append(hit / (idx + 1))
    return float(np.sum(precisions) / R)


def ndcg_at_k(sorted_rels: np.ndarray, k: int, R: int) -> float:
    """
    binary relevance NDCG@k
    DCG = sum_{i=1..k} rel_i / log2(i+1)
    IDCG = 最佳情況：前 min(R,k) 個都 relevant
    """
    if R <= 0:
        return 0.0
    kk = min(k, sorted_rels.size)
    if kk <= 0:
        return 0.0

    # DCG
    rel_k = sorted_rels[:kk].astype(np.float32)
    denom = np.log2(np.arange(2, kk + 2, dtype=np.float32))
    dcg = float(np.sum(rel_k / denom))

    # IDCG
    ideal_hits = min(R, kk)
    ideal_rel = np.ones(ideal_hits, dtype=np.float32)
    ideal_denom = np.log2(np.arange(2, ideal_hits + 2, dtype=np.float32))
    idcg = float(np.sum(ideal_rel / ideal_denom))

    return (dcg / idcg) if idcg > 0 else 0.0


def main():
    if not VEC_PATH.exists():
        raise FileNotFoundError(f"找不到向量檔：{VEC_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"找不到 gold 檔：{GOLD_PATH}")

    data = np.load(VEC_PATH)
    table_ids = data["table_ids"].astype(np.int64)
    table_vecs = data["vecs"].astype(np.float32)

    # normalize
    table_vecs /= (np.linalg.norm(table_vecs, axis=1, keepdims=True) + 1e-12)
    id_to_pos = {int(tid): i for i, tid in enumerate(table_ids)}

    gold_items = load_gold_queries(GOLD_PATH)
    if not gold_items:
        raise RuntimeError("gold query.jsonl 讀不到任何問題（或 ground_truth_list 都為空）。")

    gold_questions = [x["question"] for x in gold_items]
    gold_vecs = embed_texts(MODEL_NAME, gold_questions, BATCH_SIZE, DEVICE)

    ks = tuple(sorted(set(int(x) for x in TOP_KS)))
    if any(k <= 0 for k in ks):
        raise ValueError(f"TOP_KS 必須都是正整數：{ks}")
    max_k = max(ks)

    # ===== macro-average accumulators（統一用 ks）=====
    recall_sum = {k: 0.0 for k in ks}
    precision_sum = {k: 0.0 for k in ks}
    full_recall_sum = {k: 0.0 for k in ks}   # FullRecall@k：是否在 top-k 內找齊所有 GT（0/1）
    ndcg_sum = {k: 0.0 for k in ks}

    rr_sum = 0.0       # MRR
    ap_sum = 0.0       # MAP

    total = len(gold_items)
    eval_count = 0
    skipped_empty_gt = 0
    skipped_all_oov = 0

    try:
        from tqdm import tqdm
        it = tqdm(list(zip(gold_items, gold_vecs)), total=len(gold_items))
    except Exception:
        it = list(zip(gold_items, gold_vecs))

    for gold_obj, qvec in it:
        gt_ids_all = [int(x) for x in gold_obj["gt_ids"]]
        if not gt_ids_all:
            skipped_empty_gt += 1
            continue

        # 依 SKIP_OOV 過濾 GT
        if SKIP_OOV:
            gt_ids = [gid for gid in gt_ids_all if gid in id_to_pos]
            if not gt_ids:
                skipped_all_oov += 1
                continue
        else:
            gt_ids = gt_ids_all  # 保留 OOV（命中不了會自然拉低分數）

        gt_set = set(int(x) for x in gt_ids)
        R = len(gt_set)  # relevant 數量
        if R <= 0:
            skipped_all_oov += 1
            continue

        eval_count += 1

        sims = np.dot(table_vecs, qvec)

        # 取 top max_k ids（用於 Recall/Precision/FullRecall 的統一 ks）
        m = min(max_k, len(sims))
        top_idx = np.argpartition(-sims, kth=m - 1)[:m]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        top_ids = table_ids[top_idx].astype(np.int64)

        # top_ids 的 relevance（binary）與累積命中
        top_rels = np.fromiter((1 if int(tid) in gt_set else 0 for tid in top_ids), dtype=np.int8)
        top_cum_hits = np.cumsum(top_rels, dtype=np.int32)

        for k in ks:
            kk = min(k, top_ids.size)
            if kk <= 0:
                continue

            hit = int(top_cum_hits[kk - 1]) if kk - 1 < top_cum_hits.size else int(top_cum_hits[-1])

            # Recall@k (macro)
            recall_sum[k] += (hit / R)

            # Precision@k (macro)
            precision_sum[k] += (hit / k)

            # FullRecall@k (macro): 是否在 top-k 內把所有 GT 找齊
            full_recall_sum[k] += (1.0 if hit == R else 0.0)

        # ===== 以下指標（MRR/MAP/NDCG@k）用完整排序的 relevance 序列 =====
        # 若表數非常大而你要加速，我可以再幫你改成只算到 max_k + 用 partial rank 做 NDCG@k
        order = np.argsort(-sims)  # 全排序
        ranked_ids = table_ids[order].astype(np.int64)
        rels = np.fromiter((1 if int(tid) in gt_set else 0 for tid in ranked_ids), dtype=np.int8)

        # MRR：第一個 relevant 的 rank
        first = np.argmax(rels) if np.any(rels) else -1
        rr_sum += (1.0 / (first + 1)) if first >= 0 else 0.0

        # MAP：多正解
        ap_sum += average_precision(rels, R)

        # NDCG@k（統一 ks）
        for k in ks:
            ndcg_sum[k] += ndcg_at_k(rels, k, R)

    if eval_count == 0:
        print("沒有任何 query 被納入評估（可能 ground truth 全部不在向量庫）。")
        return

    # ===== 輸出 =====
    print(f"\nfilename: {FILENAME}")
    print("===== 評估結果 =====")
    print(f"total : {total}")
    print(f"eval_count : {eval_count}")
    if skipped_empty_gt:
        print(f"skipped_empty_gt : {skipped_empty_gt}")
    if SKIP_OOV and skipped_all_oov:
        print(f"skipped_all_oov : {skipped_all_oov}")

    print()

    for k in ks: print(f"Recall@{k} : {recall_sum[k] / eval_count:.4f}")
    print()
    for k in ks: print(f"Precision@{k} : {precision_sum[k] / eval_count:.4f}")
    print()
    for k in ks: print(f"FullRecall@{k} : {full_recall_sum[k] / eval_count:.4f}")
    print()
    for k in ks: print(f"NDCG@{k} : {ndcg_sum[k] / eval_count:.4f}")
    print()
    
    mrr = rr_sum / eval_count
    mapk = ap_sum / eval_count
    print(f"MRR@k : {mrr:.4f}")
    print(f"MAP@k : {mapk:.4f}")
    print("====================")


if __name__ == "__main__":
    main()
