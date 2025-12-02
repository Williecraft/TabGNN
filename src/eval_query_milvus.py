import json
from pathlib import Path
from typing import List, Dict
import numpy as np

# ========= 使用者可調變數 =========
SOURCE = "test/feta"
FILENAME = "query.jsonl"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
DEVICE = "cuda"                    # "cuda", "cuda:0", 或 None
TOP_KS = (1, 5, 10)

REPORT_CSV = Path("/user_data/TabGNN/results/similarity_eval_report.csv")   # Top-10 明細
RANK_SUMMARY_CSV = Path("/user_data/TabGNN/results/rank_summary.csv")       # 每筆 best_rank 摘要

# ---- Milvus 相關設定 ----
# Lite 模式不再需要 host/port，但先保留不使用
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "feta_query_corpus"

MILVUS_METRIC_TYPE = "IP"  # 因為我們用 normalize_embeddings=True，點積 = cosine，相似度越大越好
MILVUS_INDEX_TYPE = "IVF_FLAT"
MILVUS_INDEX_NLIST = 1024
MILVUS_SEARCH_NPROBE = 16
# =================================

GEN_PATH = Path(f"/user_data/TabGNN/data/generated/{SOURCE}/{FILENAME}")
GOLD_PATH = Path(f"/user_data/TabGNN/data/table/{SOURCE}/query.jsonl")


def load_generate_queries(path: Path) -> List[Dict]:
    """
    載入 generate_query.jsonl 並展開為：
    [{table_id:int, question:str}, ...]
    """
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            table_id = obj.get("table_id", obj.get("id"))
            if table_id is None:
                continue
            for q in obj.get("questions", []):
                q = (q or "").strip()
                if q:
                    items.append({"table_id": int(table_id), "question": q})
    return items


def load_gold_queries(path: Path) -> List[Dict]:
    """
    載入 query.jsonl 為：
    [{id:int, question:str}, ...]
    """
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


def build_raw_table_id_set(path: Path) -> set:
    """
    從 generate_query.jsonl 取得所有原始 table_id（不論 questions 是否為空）
    用來判斷 gold.id 是否「存在於 gen」（即使尚未產生任何 query）。
    """
    raw_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("table_id", obj.get("id"))
            if tid is None:
                continue
            try:
                raw_ids.add(int(str(tid).strip()))
            except Exception:
                pass
    return raw_ids


def embed_texts(model_name: str, texts: list, batch_size: int = 64, device: str = None) -> np.ndarray:
    """
    SentenceTransformer 向量；normalize_embeddings=True -> L2 正規化，點積即 cosine
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embs


# =============== Milvus 相關函式 ===============

def init_milvus_collection(dim: int, num_entities: int):
    """
    使用 Milvus Lite（embedded）建立 / 重建 collection，schema:
    - pk: INT64, primary key, 對應到 corpus 的 row index
    - table_id: INT64, 對應到該向量所屬的 table id
    - emb: FLOAT_VECTOR[dim]

    ★ 不再連 127.0.0.1:19530，而是直接用本機檔案 milvus_eval.db。
    """
    from pymilvus import (
        connections,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        utility,
    )

    # ★ 關鍵：使用 Lite 模式，在當前目錄建立一個 milvus_eval.db 檔案
    connections.connect(
        alias="default",
        uri="milvus_eval.db",  # 可以改成絕對路徑，例如 "/user_data/TabGNN/milvus_eval.db"
    )

    # 如果 collection 已存在就直接刪除重建（避免殘留舊資料）
    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)

    # 定義 schema
    fields = [
        FieldSchema(
            name="pk",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="table_id",
            dtype=DataType.INT64,
            is_primary=False,
        ),
        FieldSchema(
            name="emb",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
    ]
    schema = CollectionSchema(
        fields=fields,
        description=f"FETA generate_query corpus, {num_entities} entities.",
    )

    collection = Collection(name=MILVUS_COLLECTION, schema=schema)

    # 建 index（先不 load，等 insert 後再 load）
    index_params = {
        "metric_type": MILVUS_METRIC_TYPE,
        "index_type": MILVUS_INDEX_TYPE,
        "params": {
            "nlist": MILVUS_INDEX_NLIST,
        },
    }
    collection.create_index(field_name="emb", index_params=index_params)

    return collection


def insert_corpus_to_milvus(collection, corpus_vecs: np.ndarray, corpus_table_ids: np.ndarray):
    """
    把語料庫向量與對應 table_id 塞進 Milvus。
    pk = 語料庫 row index。
    """
    from pymilvus import Collection

    assert isinstance(collection, Collection)

    num = len(corpus_vecs)
    print(f"Milvus: 插入語料庫向量，共 {num} 筆...")

    pks = list(range(num))
    table_ids = corpus_table_ids.tolist()
    vectors = corpus_vecs.tolist()

    data = [pks, table_ids, vectors]
    collection.insert(data)
    collection.flush()

    # load 進記憶體，後面才能 search
    collection.load()

    print("Milvus: 插入完成並已 load collection。")


def milvus_search(collection, query_vec: np.ndarray, limit: int):
    """
    對單一 query_vec 做 Milvus search。
    回傳 hits (list of hits)，每個 hit 有 .id (pk) 和 .distance (相似度或距離)。
    """
    search_params = {
        "metric_type": MILVUS_METRIC_TYPE,
        "params": {
            "nprobe": MILVUS_SEARCH_NPROBE,
        },
    }

    # Milvus 接受的是 list[list[float]]，代表多筆 query
    data = [query_vec.tolist()]

    # 注意：limit 不可以超過 collection 中的 entity 數
    results = collection.search(
        data=data,
        anns_field="emb",
        param=search_params,
        limit=limit,
        output_fields=["table_id"],  # 若之後想要直接從 entity 拿 table_id 可用
    )

    # results 是 list[Hits]；我們只有一個 query，所以拿 results[0]
    return results[0]


# =============================================


def main():
    # 基本檢查
    if not GEN_PATH.exists():
        raise FileNotFoundError(f"找不到 generate 檔案：{GEN_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"找不到 gold 檔案：{GOLD_PATH}")

    print("載入資料中...")
    gen_items = load_generate_queries(GEN_PATH)   # 作為語料庫（展開後）
    gold_items = load_gold_queries(GOLD_PATH)     # 作為查詢
    raw_table_id_set = build_raw_table_id_set(GEN_PATH)  # 原始 table_id 集合（不管 questions 是否為空）

    if not gold_items:
        raise RuntimeError("query.jsonl 讀不到任何問題。")

    # 構建語料庫（gen 展開）
    corpus_questions = [x["question"] for x in gen_items]
    corpus_table_ids = np.array([x["table_id"] for x in gen_items], dtype=np.int64)
    corpus_questions_text = np.array(corpus_questions, dtype=object)  # 供 CSV 輸出

    print(f"語料庫問題數（gen 展開）: {len(corpus_questions)}")
    print(f"查詢問題數（gold）     : {len(gold_items)}")

    if len(corpus_questions) == 0:
        print("gen 展開後沒有任何問題（questions 全為空）— 無法評估。")
        return

    print(f"\n載入/產生向量（模型：{MODEL_NAME}）...")
    corpus_vecs = embed_texts(MODEL_NAME, corpus_questions, batch_size=BATCH_SIZE, device=DEVICE)

    # ---- 建立並填入 Milvus ----
    dim = corpus_vecs.shape[1]
    num_corpus = len(corpus_vecs)

    print(f"\n初始化 Milvus collection（dim={dim}, num={num_corpus}）...")
    from pymilvus import Collection  # 為了型別提示
    collection = init_milvus_collection(dim=dim, num_entities=num_corpus)
    insert_corpus_to_milvus(collection, corpus_vecs, corpus_table_ids)

    # 進度條
    try:
        from tqdm import tqdm
        progress = tqdm
    except Exception:
        def progress(x, **kwargs):
            return x

    ks = tuple(sorted(set(TOP_KS)))
    hit_counts = {k: 0 for k in ks}
    evaluated = 0  # 成功納入評估（gold_id 存在於 gen 原始列表）的 gold 數

    # 用於平均 rank 的統計
    per_query_best_rank = []  # 存 int 或 None（NA）
    num_with_rank = 0         # 有可計算 rank 的筆數
    sum_rank_r = 0.0          # 1/rank 的總和（用於 MRR）

    # gold 一次 encode
    gold_questions = [x["question"] for x in gold_items]
    gold_vecs = embed_texts(MODEL_NAME, gold_questions, batch_size=BATCH_SIZE, device=DEVICE)

    # 寫出 Top-10 明細
    import csv
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    RANK_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("\n評估中（gold 作查詢；Milvus Top-K 是否含相同 id 的 gen.table_id；並計算 best_rank）...")
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as fp_detail, \
         RANK_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fp_rank:
        writer_detail = csv.writer(fp_detail)
        writer_rank = csv.writer(fp_rank)

        writer_detail.writerow(["gold_id", "gold_question", "rank", "candidate_table_id", "candidate_question", "cosine_similarity"])
        writer_rank.writerow(["gold_id", "gold_question", "best_rank"])  # best_rank 可為 NA

        max_k = max(max(ks), 10)  # 至少 Top-10 便於檢查
        N = num_corpus

        # 為了讓 best_rank 和原本版本意義相近，
        # 這裡直接把 limit 設成 N，等於用 Milvus 排出全量排序。
        # 如果 N 很大，可以改成較小的 limit（例如 1000）做近似。
        full_limit = N

        for (gold_obj, qvec) in progress(zip(gold_items, gold_vecs), total=len(gold_items)):
            gold_id = gold_obj["id"]

            # 若此 gold_id 不在 gen 的原始 table_id 集合（不論有無 questions），略過（不列入 evaluated）
            if gold_id not in raw_table_id_set:
                per_query_best_rank.append(None)
                writer_rank.writerow([gold_id, gold_obj["question"], "NA"])
                continue

            evaluated += 1

            # ---- Milvus search: 取全量排序，用來計算 best_rank 以及 Top-K ----
            hits = milvus_search(collection, qvec, limit=full_limit)
            if len(hits) == 0:
                # 沒找到任何東西（理論上不會發生），當作 NA
                per_query_best_rank.append(None)
                writer_rank.writerow([gold_id, gold_obj["question"], "NA"])
                continue

            # 先把所有 hits 的 table_id 拉出來（依排序）
            # pk = hit.id 對應到 corpus 的 row index
            all_table_ids = [int(corpus_table_ids[hit.id]) for hit in hits]

            # ---- best_rank = 第一個 table_id == gold_id 的名次（1-based）----
            best_rank = None
            for idx, tid in enumerate(all_table_ids, start=1):
                if tid == gold_id:
                    best_rank = idx
                    break

            if best_rank is not None:
                per_query_best_rank.append(best_rank)
                num_with_rank += 1
                sum_rank_r += 1.0 / best_rank
                writer_rank.writerow([gold_id, gold_obj["question"], best_rank])
            else:
                per_query_best_rank.append(None)
                writer_rank.writerow([gold_id, gold_obj["question"], "NA"])

            # ---- Top-K Recall 計算（只看前 max_k）----
            top_table_ids = all_table_ids[:max_k]
            for k in ks:
                if gold_id in top_table_ids[:k]:
                    hit_counts[k] += 1

            # ---- Top-10 明細輸出 ----
            out_m = min(10, len(hits))
            for rank, hit in enumerate(hits[:out_m], start=1):
                row_idx = hit.id
                writer_detail.writerow([
                    gold_id,
                    gold_obj["question"],
                    rank,
                    int(corpus_table_ids[row_idx]),
                    corpus_questions_text[row_idx],
                    float(hit.distance),  # IP 的話可視為「相似度」
                ])

    # ----- 輸出 Recall 與平均 rank -----
    if evaluated == 0:
        print("\n沒有任何 gold 被評估（可能 gen 尚未涵蓋任何對應 id）。")
        return

    print("\n===== 結果 =====")
    for k in ks:
        r = hit_counts[k] / evaluated
        print(f"Recall@{k}: {hit_counts[k]}/{evaluated} = {r:.4f}")

    if num_with_rank > 0:
        mrr_hit = sum_rank_r / num_with_rank      # 只對有命中的樣本做平均
        mrr_overall = sum_rank_r / evaluated      # 把沒命中的也算進母數
        print(f"\nMRR (Hits)\t：{mrr_hit:.4f}"
              f"\nMRR (overall)\t：{mrr_overall:.4f}"
              f"\n(樣本數 {num_with_rank} / 評估 {evaluated})")
    else:
        print("\nMRR：無法計算（所有可評估 gold 在 corpus 中都沒有對應 id 的生成 query）")


if __name__ == "__main__":
    main()
