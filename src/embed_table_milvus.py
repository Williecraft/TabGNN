# embed_table_milvus.py
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# ========= 可調參數 =========
SOURCE = "test/feta"
FILENAME = "query_5lines.jsonl"

TABLE_PATH = Path(f"/user_data/TabGNN/data/table/{SOURCE}/table.jsonl")
GEN_PATH = Path(f"/user_data/TabGNN/data/generated/{SOURCE}/{FILENAME}")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
DEVICE = "cuda"

# Milvus Lite（本地）
MILVUS_DIR = Path(f"/user_data/TabGNN/data/milvus/{SOURCE}")
MILVUS_DIR.mkdir(parents=True, exist_ok=True)
MILVUS_URI = str(MILVUS_DIR / FILENAME.replace(".jsonl", ".db"))

COLLECTION_NAME = "table_vectors"
DIM = 1024  # bge-m3
# ===========================


# ---------- json2csv（你已驗證穩定的版本） ----------
def json2csv(jtable: dict, top=None) -> str:
    import csv
    import pandas as pd

    def parse_csv_row(s: str) -> list[str]:
        return next(csv.reader([s]))

    header = jtable.get("header", [])
    if isinstance(header, str):
        header = parse_csv_row(header)
    elif isinstance(header, list):
        if len(header) == 1 and isinstance(header[0], str) and "," in header[0]:
            header = parse_csv_row(header[0])
        else:
            header = [("" if h is None else str(h)) for h in header]

    rows = jtable.get("instances", [])
    if top is not None:
        rows = rows[:int(top)]

    parsed_rows = []
    for r in rows:
        if isinstance(r, str):
            parsed_rows.append(parse_csv_row(r))
        elif isinstance(r, list):
            parsed_rows.append([("" if x is None else str(x)) for x in r])

    df = pd.DataFrame(parsed_rows, columns=header)
    return df.to_csv(index=False)


# ---------- util ----------
def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_gen_map(path: Path) -> Dict[int, Dict]:
    mp = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tid = obj.get("table_id", obj.get("id"))
            if tid is not None:
                mp[int(tid)] = obj
    return mp


def build_table_text(table_obj: Dict, gen_obj: Dict) -> str:
    parts = [
        f"TABLE_ID: {table_obj['id']}",
        f"FILE: {table_obj.get('file_name','')}",
        f"SHEET: {table_obj.get('sheet_name','')}",
        "TABLE_CSV:",
        json2csv(table_obj, top=None),
        "GENERATED_QUESTIONS:",
    ]
    for q in gen_obj.get("questions", []):
        parts.append(f"- {q}")
    return "\n".join(parts)


# ---------- main ----------
def main():
    print("🔌 Connect Milvus Lite")
    connections.connect("default", uri=MILVUS_URI)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="table_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="Table-level vectors")
    collection = Collection(COLLECTION_NAME, schema)

    tables = load_jsonl(TABLE_PATH)
    gen_map = load_gen_map(GEN_PATH)

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    table_ids, vecs = [], []

    print("🔧 Embedding tables...")
    for t in tables:
        tid = int(t["id"])
        if tid not in gen_map:
            continue
        text = build_table_text(t, gen_map[tid])
        emb = model.encode(text, normalize_embeddings=True)
        table_ids.append(tid)
        vecs.append(emb.tolist())

    print(f"📥 Insert {len(vecs)} tables")
    collection.insert([table_ids, vecs])

    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "FLAT"},
    )
    collection.load()

    print("✅ Milvus table vector DB ready")


if __name__ == "__main__":
    main()
