from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from typing import List

_embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")


def _embed_texts(texts):
    
    embeddings = _embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def transform_queries_to_embeddings(input_path: str, output_path: str) -> None:
    
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                
                continue

            queries: List[str] = obj.get('queries', []) or []
            if queries:
                vecs = _embed_texts(queries)
                
                obj['queries'] = vecs.tolist()
            else:
                obj['queries'] = []

            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write('\n')


if __name__ == "__main__":
    default_input = r"data\train\feta\generate_query.jsonl"
    default_output = r"data\train\feta\embeddings_query.jsonl"
    transform_queries_to_embeddings(default_input, default_output)


