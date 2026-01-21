# embed_table.py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# ========= 可調參數 =========
SOURCE = "train/spider_multitabqa"
# 第一步產出的檔案
GEN_FILENAME = "query_ollama.jsonl"

TABLE_PATH = Path(f"/user_data/TabGNN/data/table/{SOURCE}/table.jsonl")
GEN_PATH = Path(f"/user_data/TabGNN/data/generated/{SOURCE}/{GEN_FILENAME}")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
DEVICE = "cuda"  # "cuda" / "cuda:0" / None
TOPROW = 10

OUT_DIR = Path(f"/user_data/TabGNN/data/embeddings/{SOURCE}")
OUT_PATH = OUT_DIR / f"{GEN_FILENAME.replace('.jsonl','')}.npz"
# ===========================

def json2csv(jtable: dict, top=None) -> str:
    import csv
    import pandas as pd

    def parse_csv_row(s: str) -> list[str]:
        return next(csv.reader([s]))

    header = jtable.get("header", [])
    if isinstance(header, str):
        header = parse_csv_row(header)
    elif isinstance(header, list):
        if len(header) == 1 and isinstance(header[0], str) and ("," in header[0]):
            header = parse_csv_row(header[0])
        else:
            header = [("" if h is None else str(h)) for h in header]
    else:
        header = [str(header)]

    instances = jtable.get("instances", [])
    rows = instances if top is None else instances[:int(top)]

    parsed_rows = []
    for r in rows:
        if isinstance(r, str):
            parsed_rows.append(parse_csv_row(r))
        elif isinstance(r, list):
            parsed_rows.append([("" if x is None else str(x)) for x in r])
        else:
            parsed_rows.append([str(r)])

    ncol = len(header)
    fixed_rows = []
    for row in parsed_rows:
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))
        elif len(row) > ncol:
            row = row[:ncol]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)
    return df.to_csv(index=False)

def embed_texts(model_name: str, texts: List[str], batch_size: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # dot = cosine
    )
    return embs.astype(np.float32)

def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_gen_map(path: Path) -> Dict[int, Dict]:
    mp = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("table_id", obj.get("id"))
            if tid is None:
                continue
            mp[int(tid)] = obj
    return mp

def build_table_text(table_obj: Dict, gen_obj: Dict) -> str:
    tid = int(table_obj["id"])
    file_name = table_obj.get("file_name", "")
    sheet_name = table_obj.get("sheet_name", "")

    headers = gen_obj.get("headers", []) if isinstance(gen_obj, dict) else []
    questions = gen_obj.get("questions", []) if isinstance(gen_obj, dict) else []

    full_csv = json2csv(table_obj, top=TOPROW)

    parts = []
    parts.append(f"TABLE_ID: {tid}")
    if file_name:
        parts.append(f"FILE: {file_name}")
    if sheet_name:
        parts.append(f"SHEET: {sheet_name}")

    if headers:
        parts.append("HEADERS:")
        for h in headers:
            parts.append(f"- {str(h)}")

    parts.append("TABLE_CSV_FULL:")
    parts.append(full_csv)

    if questions:
        parts.append("GENERATED_QUESTIONS:")
        for q in questions:
            q = (q or "").strip()
            if q:
                parts.append(f"- {q}")

    return "\n".join(parts)

def main():
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"找不到 table.jsonl：{TABLE_PATH}")
    if not GEN_PATH.exists():
        raise FileNotFoundError(f"找不到生成 query 檔：{GEN_PATH}")

    print("Loading table.jsonl...")
    tables = load_jsonl(TABLE_PATH)
    print("Loading generated query jsonl...")
    gen_map = load_gen_map(GEN_PATH)

    table_ids = []
    table_texts = []

    for t in tables:
        tid = int(t.get("id"))
        gen_obj = gen_map.get(tid)
        if gen_obj is None:
            continue  # 沒生成就不做向量（也可改成仍做）
        table_ids.append(tid)
        table_texts.append(build_table_text(t, gen_obj))

    if not table_texts:
        raise RuntimeError("沒有任何可向量化的 table（gen_map 對不上 table.jsonl）。")

    print(f"Embedding tables: {len(table_texts)}")
    vecs = embed_texts(MODEL_NAME, table_texts, BATCH_SIZE, DEVICE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, table_ids=np.array(table_ids, dtype=np.int64), vecs=vecs)
    print(f"Saved NPZ: {OUT_PATH}")

if __name__ == "__main__":
    main()
