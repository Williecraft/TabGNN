# eval_query_vec.py
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

# ========= 可調參數 =========
SOURCE = "test/feta"
FILENAME = "query_5lines.npz"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
DEVICE = "cuda"   # "cuda" / "cuda:0" / None
TOP_KS = (1, 5, 10)

VEC_PATH = Path(f"/user_data/TabGNN/data/embeddings/{SOURCE}/{FILENAME}")
GOLD_PATH = Path(f"/user_data/TabGNN/data/table/{SOURCE}/query.jsonl")

SKIP_OOV = True  # gold_id 不在向量庫時是否跳過
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
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            if q and ("id" in obj):
                items.append({"id": int(obj["id"]), "question": q})
    return items

def ndcg_single(rank: int, k: int) -> float:
    """
    單一正解 NDCG@k（只有一個 relevant table）：
    rank<=k => 1/log2(rank+1), else 0
    """
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)

def main():
    if not VEC_PATH.exists():
        raise FileNotFoundError(f"找不到向量檔：{VEC_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"找不到 gold 檔：{GOLD_PATH}")

    data = np.load(VEC_PATH)
    table_ids = data["table_ids"].astype(np.int64)
    table_vecs = data["vecs"].astype(np.float32)

    # 保險 normalize
    table_vecs /= (np.linalg.norm(table_vecs, axis=1, keepdims=True) + 1e-12)
    id_to_pos = {int(tid): i for i, tid in enumerate(table_ids)}

    gold_items = load_gold_queries(GOLD_PATH)
    if not gold_items:
        raise RuntimeError("gold query.jsonl 讀不到任何問題。")

    gold_questions = [x["question"] for x in gold_items]
    gold_vecs = embed_texts(MODEL_NAME, gold_questions, BATCH_SIZE, DEVICE)

    ks = tuple(sorted(set(TOP_KS)))
    hit_counts = {k: 0 for k in ks}

    rr_sum = 0.0
    ap_sum = 0.0  # 單一正解時 AP = 1/rank，因此 MAP = MRR（但仍照樣計）
    ndcg10_sum = 0.0

    total = len(gold_items)
    eval_count = 0
    skipped_oov = 0

    # progress（可省略，但保留）
    try:
        from tqdm import tqdm
        it = tqdm(list(zip(gold_items, gold_vecs)), total=len(gold_items))
    except Exception:
        it = list(zip(gold_items, gold_vecs))

    max_k = max(ks)

    for gold_obj, qvec in it:
        gold_id = int(gold_obj["id"])

        if gold_id not in id_to_pos:
            if SKIP_OOV:
                skipped_oov += 1
                continue
            else:
                eval_count += 1
                continue

        eval_count += 1

        sims = np.dot(table_vecs, qvec)
        pos = id_to_pos[gold_id]
        score_gold = sims[pos]

        # best-rank tie handling
        rank = int(np.sum(sims > score_gold)) + 1

        # MRR / MAP（單一正解）
        rr = 1.0 / rank
        rr_sum += rr
        ap_sum += rr

        # Top-K
        m = min(max_k, len(sims))
        top_idx = np.argpartition(-sims, kth=m - 1)[:m]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        top_ids = table_ids[top_idx]

        for k in ks:
            if gold_id in top_ids[:k]:
                hit_counts[k] += 1

        # NDCG@10（照你截圖）
        ndcg10_sum += ndcg_single(rank, 10)

    if eval_count == 0:
        print("沒有任何 query 被納入評估（可能向量庫缺少對應 table_id）。")
        return

    # ===== 輸出（照你截圖格式）=====
    print(f"\nfilename: {FILENAME}")
    print("===== 評估結果 =====")
    print(f"總查詢數 : {total}")
    print(f"可對齊評估的查詢數 : {eval_count}")
    print(f"total : {total}")
    print(f"eval_count : {eval_count}")

    # Recall@k
    for k in ks:
        print(f"Recall@{k} : {hit_counts[k] / eval_count:.4f}")

    # MRR / MAP / NDCG
    mrr = rr_sum / eval_count
    mapk = ap_sum / eval_count
    ndcg10 = ndcg10_sum / eval_count

    print(f"MRR@k : {mrr:.4f}")
    print(f"MAP@k : {mapk:.4f}")
    print(f"NDCG@10 : {ndcg10:.4f}")
    print("====================")

if __name__ == "__main__":
    main()
