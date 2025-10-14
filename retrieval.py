import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
_embedding_model = SentenceTransformer(_MODEL_NAME)


def _embed_query(text: str) -> np.ndarray:
    vec = _embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0]


def _iter_tables(embeddings_path: Path):
    with embeddings_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj


def _similarity_to_table(query_vec: np.ndarray, table_obj: Dict[str, Any]) -> float:
    queries: List[List[float]] = table_obj.get("queries", []) or []
    if not queries:
        return -1.0
    mat = np.asarray(queries, dtype=np.float32)
    
    sims = mat @ query_vec.astype(np.float32)
    return float(np.max(sims))


def retrieve_top_k(query: str, embeddings_path: str, top_k: int = 8) -> List[Dict[str, Any]]:
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    query_vec = _embed_query(query)
    results = []
    for obj in _iter_tables(path):
        score = _similarity_to_table(query_vec, obj)
        results.append({
            "table_id": obj.get("table_id"),
            "table_name": obj.get("table_name"),
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def main():
    
    query = input("輸入query：").strip()

    top = retrieve_top_k(query=query, embeddings_path=r"c:\\Users\\user\\Desktop\\TableRetrieval\\TabGNN\\data\\train\\feta\\embeddings_query.jsonl", top_k=8)
    
    for i, item in enumerate(top, start=1):
        print(f"{i}. table_id={item['table_id']} | table_name={item['table_name']} | score={item['score']:.4f}")


    print("\nJSON:")
    print(json.dumps(top, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()