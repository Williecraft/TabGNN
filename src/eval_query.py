import json
from pathlib import Path
from typing import List, Dict
import numpy as np

# ========= 使用者可調變數 =========
SOURCE = "test/feta"

MODEL_NAME = "BAAI/bge-m3"  
BATCH_SIZE = 32
DEVICE = "cuda"                    # "cuda", "cuda:0", 或 None
TOP_KS = (1, 5, 10)

REPORT_CSV = Path("/user_data/TabGNN/results/similarity_eval_report.csv")   # Top-10 明細
RANK_SUMMARY_CSV = Path("/user_data/TabGNN/results/rank_summary.csv")       # 每筆 best_rank 摘要
# =================================

GEN_PATH = Path("/user_data/TabGNN/data/test/feta/generate_query.jsonl")
GOLD_PATH = Path("/user_data/TabGNN/data/table/test/feta/query.jsonl")

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
            except:
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

    print(f"\n載入/產生向量（模型：{MODEL_NAME}）...")
    if len(corpus_questions) == 0:
        print("gen 展開後沒有任何問題（questions 全為空）— 無法評估。")
        return

    corpus_vecs = embed_texts(MODEL_NAME, corpus_questions, batch_size=BATCH_SIZE, device=DEVICE)

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
    num_with_rank = 0         # 有可計算 rank 的筆數（corpus 中存在該 table_id 的 query）
    sum_rank_r = 0              # rank 總和

    # gold 一次 encode
    gold_questions = [x["question"] for x in gold_items]
    gold_vecs = embed_texts(MODEL_NAME, gold_questions, batch_size=BATCH_SIZE, device=DEVICE)

    # 寫出 Top-10 明細
    import csv
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    RANK_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("\n評估中（gold 作查詢；Top-K 是否含相同 id 的 gen.table_id；並計算 best_rank）...")
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as fp_detail, \
         RANK_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fp_rank:
        writer_detail = csv.writer(fp_detail)
        writer_rank = csv.writer(fp_rank)

        writer_detail.writerow(["gold_id", "gold_question", "rank", "candidate_table_id", "candidate_question", "cosine_similarity"])
        writer_rank.writerow(["gold_id", "gold_question", "best_rank"])  # best_rank 可為 NA

        max_k = max(max(ks), 10)  # 至少 Top-10 便於檢查
        N = len(corpus_vecs)

        for (gold_obj, qvec) in progress(zip(gold_items, gold_vecs), total=len(gold_items)):
            gold_id = gold_obj["id"]

            # 若此 gold_id 不在 gen 的原始 table_id 集合（不論有無 questions），略過（不列入 evaluated）
            if gold_id not in raw_table_id_set:
                # print(f"跳過 gold_id={gold_id}（gen 尚未產生此表）。")
                per_query_best_rank.append(None)
                writer_rank.writerow([gold_id, gold_obj["question"], "NA"])
                continue

            evaluated += 1
            sims = np.dot(corpus_vecs, qvec)  # (N,)

            # ---- 計算 best_rank（即使超過 Top-K）----
            mask_same_id = (corpus_table_ids == gold_id)
            if np.any(mask_same_id):
                max_sim_same = np.max(sims[mask_same_id])
                # 名次 = 1 + 比它更大的相似度數量（處理並列時給最好的名次）
                better = int(np.sum(sims > max_sim_same))
                best_rank = better + 1
                per_query_best_rank.append(best_rank)
                num_with_rank += 1
                sum_rank_r += 1/best_rank
                writer_rank.writerow([gold_id, gold_obj["question"], best_rank])
            else:
                # gen 有這個 id（raw 中存在），但 corpus（展開）沒有任何該 id 的問題 → 無法算 rank
                per_query_best_rank.append(None)
                writer_rank.writerow([gold_id, gold_obj["question"], "NA"])

            # ---- Top-K 明細（原本邏輯）----
            m = min(max_k, N)
            top_idx = np.argpartition(-sims, kth=m-1)[:m]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            top_table_ids = corpus_table_ids[top_idx]

            for k in ks:
                if gold_id in top_table_ids[:k]:
                    hit_counts[k] += 1

            out_m = min(10, len(top_idx))
            for rank, idx in enumerate(top_idx[:out_m], start=1):
                writer_detail.writerow([
                    gold_id,
                    gold_obj["question"],
                    rank,
                    int(corpus_table_ids[idx]),
                    corpus_questions_text[idx],
                    float(sims[idx]),
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
        mrr_hit = sum_rank_r / num_with_rank
        mrr_overall = sum_rank_r / evaluated
        print(f"\nMRR (Hits)\t：{mrr_hit:.4f}"
              f"\nMRR (overall)\t：{mrr_overall:.4f}"
              f"\n(樣本數 {num_with_rank} / 評估 {evaluated})")
    else:
        print("\nMRR：無法計算（所有可評估 gold 在 corpus 中都沒有對應 id 的生成 query）")

if __name__ == "__main__":
    main()
