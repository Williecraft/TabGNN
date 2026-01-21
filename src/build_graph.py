"""建置異構圖 (HeteroData) 用於 GNN 訓練與評估"""
import json
import csv
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ========= 可調參數 =========
TABLE_JSONL_PATH = '/user_data/TabGNN/data/table/test/feta/table.jsonl'
OUTPUT_GRAPH_PATH = "/user_data/TabGNN/data/processed/graph_evaluate.pt"
MODEL_NAME = 'BAAI/bge-m3'
DEVICE = 'cuda'

# 相似度邊的 Top-K 設定
K_TABLE = 5   # 每張表連接 K 個最相似的表
K_COLUMN = 5  # 每個欄位連接 K 個最相似的欄位
# ===========================


def main():
    # --- 1. 讀取資料 ---
    tables = []
    with open(TABLE_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            tables.append(json.loads(line))
    print(f"讀取完成，共 {len(tables)} 張表格。")

    # 載入嵌入模型
    embedder = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # --- 2. 解析表格並建立節點 ---
    data = HeteroData()
    print("\n=== 解析 JSONL ===")

    table_docs = []   # 表格的文字描述
    column_docs = []  # 欄位的文字描述
    page_docs = []    # 頁面標題

    # 索引映射
    table_id_to_idx = {}
    column_global_id_to_idx = {}
    page_title_to_idx = {}

    # 結構邊
    edge_table_to_col_src, edge_table_to_col_dst = [], []
    edge_table_to_page_src, edge_table_to_page_dst = [], []

    # 原始資料
    meta_table_ids = []
    meta_page_titles = []
    meta_col_global_ids = []

    # 用於建立 same_page 邊
    page_to_table_idxs = {}

    for item in tqdm(tables, desc="解析表格"):
        try:
            # 解析 header
            if not item.get('header') or not item['header'][0]:
                continue
            header_list = next(csv.reader([item['header'][0]]))

            # 解析 instances
            instance_rows = [next(csv.reader([row_str])) for row_str in item.get('instances', [])]

            table_id = item['id']
            if table_id in table_id_to_idx:
                continue  # 跳過重複

            # 表格節點索引
            current_table_idx = len(table_docs)
            table_id_to_idx[table_id] = current_table_idx
            meta_table_ids.append(table_id)

            # 頁面節點
            page_title = item.get('metadata', {}).get('table_page_title') or f"__UNKNOWN_PAGE_{table_id}__"
            if page_title not in page_title_to_idx:
                current_page_idx = len(page_docs)
                page_title_to_idx[page_title] = current_page_idx
                page_docs.append(page_title)
                meta_page_titles.append(page_title)
            else:
                current_page_idx = page_title_to_idx[page_title]

            # Table -> Page 邊
            edge_table_to_page_src.append(current_table_idx)
            edge_table_to_page_dst.append(current_page_idx)

            # 記錄同頁表格
            if page_title not in page_to_table_idxs:
                page_to_table_idxs[page_title] = []
            page_to_table_idxs[page_title].append(current_table_idx)

            # 表格文字描述
            table_doc = " ".join(filter(None, [
                f"Page: {page_title}",
                f"Sheet: {item.get('sheet_name', '')}",
                f"Section: {item.get('metadata', {}).get('table_section_title', '')}",
                f"Columns: {', '.join(header_list)}",
                f"Data: {'; '.join([', '.join(row) for row in instance_rows[:5]])}"
            ]))
            table_docs.append(table_doc)

            # 欄位節點
            for col_idx, col_name in enumerate(header_list):
                col_global_id = f"{table_id}::{col_name}::{col_idx}"
                if col_global_id in column_global_id_to_idx:
                    continue

                current_col_idx = len(column_docs)
                column_global_id_to_idx[col_global_id] = current_col_idx
                meta_col_global_ids.append(col_global_id)

                # Table -> Column 邊
                edge_table_to_col_src.append(current_table_idx)
                edge_table_to_col_dst.append(current_col_idx)

                # 欄位文字描述
                sample_values = [row[col_idx] for row in instance_rows[:10] if len(row) > col_idx and row[col_idx].strip()]
                col_doc = " ".join(filter(None, [
                    f"Column: {col_name}",
                    f"Belongs to table: {page_title} - {item.get('sheet_name', '')}",
                    f"Values: {', '.join(sample_values)}"
                ]))
                column_docs.append(col_doc)

        except Exception as e:
            print(f"處理表格 {item.get('id')} 時發生錯誤: {e}")
            continue

    print(f"\n--- 建圖 (共 {len(table_docs)} 表, {len(column_docs)} 欄, {len(page_docs)} 頁) ---")

    # --- 3. 計算嵌入向量 ---
    print("開始 embeddings...")
    data['table'].x = torch.tensor(embedder.encode(table_docs, show_progress_bar=True), dtype=torch.float)
    data['column'].x = torch.tensor(embedder.encode(column_docs, show_progress_bar=True), dtype=torch.float)
    data['page'].x = torch.tensor(embedder.encode(page_docs, show_progress_bar=True), dtype=torch.float)
    print("embeddings 完成。")

    # 儲存原始 ID
    data['table'].id = meta_table_ids
    data['column'].global_id = meta_col_global_ids
    data['page'].title = meta_page_titles

    # 儲存映射表
    data.metadata_maps = {
        'table_id_to_idx': table_id_to_idx,
        'column_global_id_to_idx': column_global_id_to_idx,
        'page_title_to_idx': page_title_to_idx
    }

    # --- 4. 建立結構邊 ---
    data['table', 'has_column', 'column'].edge_index = torch.tensor([edge_table_to_col_src, edge_table_to_col_dst], dtype=torch.long)
    data['table', 'comes_from', 'page'].edge_index = torch.tensor([edge_table_to_page_src, edge_table_to_page_dst], dtype=torch.long)

    # Same Page 邊
    print("正在建立同一 page 的 table 連邊 (same_page)...")
    same_src, same_dst = [], []
    for t_idxs in page_to_table_idxs.values():
        if len(t_idxs) <= 1:
            continue
        for i in range(len(t_idxs)):
            for j in range(i + 1, len(t_idxs)):
                same_src.extend([t_idxs[i], t_idxs[j]])
                same_dst.extend([t_idxs[j], t_idxs[i]])

    if same_src:
        data['table', 'same_page', 'table'].edge_index = torch.tensor([same_src, same_dst], dtype=torch.long)
        print(f"建立 same_page edges: {len(same_src)} 條（含反向）")
    else:
        print("沒有發現同一 page 下有多於 1 張 table，未建立 same_page edges。")

    # 建立反向邊
    print("正在建立反向邊...")
    data = data.to_homogeneous().to_heterogeneous()

    # --- 5. 建立相似度邊 ---
    # Table 相似度
    print("計算 'table' <-> 'table' 相似度邊...")
    xt = F.normalize(data['table'].x, p=2, dim=1)
    sim_matrix = torch.matmul(xt, xt.T)
    sim_matrix.fill_diagonal_(float('-inf'))
    _, top_k_indices = torch.topk(sim_matrix, k=min(K_TABLE, xt.size(0) - 1), dim=1)

    sim_src, sim_dst = [], []
    for i in range(xt.size(0)):
        for j in top_k_indices[i]:
            sim_src.append(i)
            sim_dst.append(j.item())
    data['table', 'similar_table', 'table'].edge_index = torch.tensor([sim_src, sim_dst], dtype=torch.long)

    # Column 相似度
    print("計算 'column' <-> 'column' 相似度邊...")
    xc = F.normalize(data['column'].x, p=2, dim=1)
    sim_matrix_c = torch.matmul(xc, xc.T)
    sim_matrix_c.fill_diagonal_(float('-inf'))
    _, top_k_indices_c = torch.topk(sim_matrix_c, k=min(K_COLUMN, xc.size(0) - 1), dim=1)

    sim_src_c, sim_dst_c = [], []
    for i in range(xc.size(0)):
        for j in top_k_indices_c[i]:
            sim_src_c.append(i)
            sim_dst_c.append(j.item())
    data['column', 'similar_content', 'column'].edge_index = torch.tensor([sim_src_c, sim_dst_c], dtype=torch.long)

    # --- 6. 儲存圖 ---
    print("\n--- 圖構建完成！ ---")
    print("\n最終圖結構 (HeteroData):")
    print(data)

    print(f"\n將圖儲存至 {OUTPUT_GRAPH_PATH}...")
    torch.save(data, OUTPUT_GRAPH_PATH)
    print("完成！")


if __name__ == "__main__":
    main()
