# eval_query_milvus.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# ========= 可調參數 =========
SOURCE = "test/feta"
FILENAME = "query_5lines.db"

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"
TOP_KS = (1, 5, 10)

MILVUS_URI = f"/user_data/TabGNN/data/milvus/{SOURCE}/{FILENAME}"
COLLECTION_NAME = "table_vectors"

GOLD_PATH = Path(f"/user_data/TabGNN/data/table/{SOURCE}/query.jsonl")
# ===========================


def ndcg_single(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)


def main():
    connections.connect("default", uri=MILVUS_URI)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    with GOLD_PATH.open("r", encoding="utf-8") as f:
        gold_items = [json.loads(l) for l in f if l.strip()]

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    hit_counts = {k: 0 for k in TOP_KS}
    rr_sum = 0.0
    ap_sum = 0.0
    ndcg10_sum = 0.0

    total = len(gold_items)
    eval_count = 0

    for g in gold_items:
        gold_id = int(g["id"])
        qvec = model.encode(g["question"], normalize_embeddings=True)

        results = collection.search(
            data=[qvec.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=max(TOP_KS),
            output_fields=["table_id"],
        )[0]

        eval_count += 1
        rank = None
        for i, hit in enumerate(results):
            if hit.entity.get("table_id") == gold_id:
                rank = i + 1
                break

        if rank is not None:
            rr = 1.0 / rank
            rr_sum += rr
            ap_sum += rr
            for k in TOP_KS:
                if rank <= k:
                    hit_counts[k] += 1
            ndcg10_sum += ndcg_single(rank, 10)

    print(f"\nfilename: {FILENAME}")
    print("With Milvus")
    print("===== 評估結果 =====")
    print(f"總查詢數 : {total}")
    print(f"可對齊評估的查詢數 : {eval_count}")
    print(f"total : {total}")
    print(f"eval_count : {eval_count}")

    for k in TOP_KS:
        print(f"Recall@{k} : {hit_counts[k] / eval_count:.4f}")

    print(f"MRR@k : {rr_sum / eval_count:.4f}")
    print(f"MAP@k : {ap_sum / eval_count:.4f}")
    print(f"NDCG@10 : {ndcg10_sum / eval_count:.4f}")
    print("====================")


if __name__ == "__main__":
    main()
