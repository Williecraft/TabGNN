# -*- coding: utf-8 -*-
"""
Single-query semantic search:
- 輸入一段 USER_QUERY
- 與 generate_query.jsonl 展開的 query 比對
- 輸出前 K 名的 (table_id, candidate_query, cosine_similarity)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# ========= 可調變數 =========
SOURCE = "test/feta"
GEN_PATH = Path(fr"data/{SOURCE}/generate_query.jsonl")

MODEL_NAME = "BAAI/bge-m3"   # 若有多語需求可用：paraphrase-multilingual-MiniLM-L12-v2
DEVICE = "cuda"                    # "cuda", "cuda:0", 或 None
BATCH_SIZE = 64                    # 嵌入語料時的 batch
TOP_K = 10                     # 預設取前 K 名
USER_QUERY = "What were the achievements of Alex Yunevich during hit football career?"  # 測試查詢
# ===========================

# ---- 資料載入 ----
def load_generate_queries(path: Path) -> List[Dict]:
    """
    讀取 generate_query.jsonl，展開為：
    [{"table_id": int, "question": str}, ...]
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

# ---- 向量化 ----
def embed_texts(model_name: str, texts: List[str], batch_size: int = 64, device: str = None) -> np.ndarray:
    """
    使用 SentenceTransformer 做向量；normalize_embeddings=True -> L2 normalize
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

# ---- 搜尋核心 ----
class GenSearch:
    def __init__(self, gen_items: List[Dict], model_name: str, device: str, batch_size: int):
        if not gen_items:
            raise RuntimeError("generate_query.jsonl 無可用的 query。")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # 語料
        self.corpus_questions: List[str] = [x["question"] for x in gen_items]
        self.corpus_table_ids: np.ndarray = np.array([x["table_id"] for x in gen_items], dtype=np.int64)

        # 預先嵌入語料
        self.corpus_vecs: np.ndarray = embed_texts(
            model_name, self.corpus_questions, batch_size=batch_size, device=device
        )

    def search_one(self, query: str, top_k: int = 10) -> List[Tuple[int, str, float]]:
        """
        對單一 query 做搜尋，回傳 [(table_id, candidate_question, cosine_similarity), ...]
        """
        if not query or not query.strip():
            return []

        qvec = embed_texts(self.model_name, [query], batch_size=1, device=self.device)[0]  # (d,)
        sims = self.corpus_vecs @ qvec  # (N,)

        k = min(top_k, len(sims))
        # 先取前 k 個索引（無序），再在這些裡面排序
        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        results = []
        for i in idx:
            results.append((
                int(self.corpus_table_ids[i]),
                self.corpus_questions[i],
                float(sims[i])
            ))
        return results

def main():
    # 讀取 gen 語料
    if not GEN_PATH.exists():
        raise FileNotFoundError(f"找不到 generate 檔案：{GEN_PATH}")
    gen_items = load_generate_queries(GEN_PATH)
    print(f"載入 gen 展開 query 數量：{len(gen_items)}")

    # 建立搜尋器
    searcher = GenSearch(gen_items, MODEL_NAME, DEVICE, BATCH_SIZE)

    # 測試一次
    print(f"\n【Query】{USER_QUERY}")
    results = searcher.search_one(USER_QUERY, top_k=TOP_K)

    # 輸出
    print(f"\nTop-{TOP_K} 相似結果：")
    for rank, (tid, cand_q, sim) in enumerate(results, start=1):
        print(f"{rank:>2}. table_id={tid} | sim={sim:.4f}\n    {cand_q}")

if __name__ == "__main__":
    main()
