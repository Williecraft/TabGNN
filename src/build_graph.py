import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, DataLoader
from torch_geometric.nn import RGCNConv
from sentence_transformers import SentenceTransformer
import random
import json
import os 
import csv 
from tqdm import tqdm 

# 異質圖教學(HeteroData): https://zhuanlan.zhihu.com/p/659971512

# 讀資料庫
tables = []
TABLE_JSONL_PATH = '/user_data/TabGNN/data/table/test/feta/table.jsonl' 
with open(TABLE_JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        temp = json.loads(line)
        tables.append(temp)
print(f"讀取完成，共 {len(tables)} 張表格。")

# 適用純英文轉句向量，如果有多種語言用 distiluse-base-multilingual-cased-v2
# embedder = SentenceTransformer('sentence-transformers/BGE-M3', device='cuda')
embedder = SentenceTransformer('BAAI/bge-m3', device='cuda')
# embedder = SentenceTransformer('BGE-M3', device='cuda')

data = HeteroData()
print("\n=== 解析 JSONL ===")

table_docs = [] # 表格的完整文字描述(後面會拼起來)
column_docs = []# 欄位的名稱、所屬表格和前 10 個值。
page_docs = [] # 維基百科頁面的標題

# 索引映射(id推回索引)
table_id_to_idx = {} 
column_global_id_to_idx = {} 
page_title_to_idx = {}

# 結構邊
edge_table_to_col_src = [] # table -> col 的起點
edge_table_to_col_dst = []  # table -> col 的終點
edge_table_to_page_src = [] # talbe -> page 的起點
edge_table_to_page_dst = [] #table -> page 的終點

# 儲存節點的原始資料
meta_table_ids = []
meta_page_titles = []
meta_col_global_ids = []

# 新增：記錄每個 page 下有哪些 table idx（用於 same_page 連邊）
page_to_table_idxs = {}

for item in tqdm(tables, desc="解析表格"):
    try:
        
        # header 原本是字串，切成一個list
        if not item.get('header') or not item['header'][0]:
            continue 
        header_str = item['header'][0]
        header_list = next(csv.reader([header_str])) 

        # instance 同理
        instance_rows = []
        for row_str in item.get('instances', []):
            instance_rows.append(next(csv.reader([row_str])))

        table_id = item['id']
        # 資料庫中好像有重複的
        if table_id in table_id_to_idx:
            continue 
        
        # 賦予這張表一個新的節點索引 (0, 1, 2...)
        current_table_idx = len(table_docs)
        table_id_to_idx[table_id] = current_table_idx
        meta_table_ids.append(table_id)

        # Page
        page_title = item.get('metadata', {}).get('table_page_title')
        if not page_title:
            page_title = f"__UNKNOWN_PAGE_{table_id}__" 
        
        if page_title not in page_title_to_idx:
            # 如果是新的 Page 就登記到page_docs
            current_page_idx = len(page_docs)
            page_title_to_idx[page_title] = current_page_idx
            page_docs.append(page_title) 
            meta_page_titles.append(page_title)
        else:
            # 舊的就找出索引
            current_page_idx = page_title_to_idx[page_title]

        # 登記 table -> page 的結構邊 ('Table', 'comes_from', 'Page') 
        edge_table_to_page_src.append(current_table_idx)
        edge_table_to_page_dst.append(current_page_idx)

        # 同步記錄 page -> table idx 列表（後面用來建立 same_page edges）
        if page_title not in page_to_table_idxs:
            page_to_table_idxs[page_title] = []
        page_to_table_idxs[page_title].append(current_table_idx)

        # Table 的簡述
        table_doc_parts = [
            f"Page: {page_title}",
            f"Sheet: {item.get('sheet_name', '')}",
            f"Section: {item.get('metadata', {}).get('table_section_title', '')}",
            f"Columns: {', '.join(header_list)}",
            f"Data: {'; '.join([', '.join(row) for row in instance_rows[:5]])}"
        ]
        table_docs.append(" ".join(filter(None, table_doc_parts)))

        for col_idx, col_name in enumerate(header_list):
            # 建立一個全局唯一的欄位 ID
            col_global_id = f"{table_id}::{col_name}::{col_idx}"
            if col_global_id in column_global_id_to_idx:
                continue
            
            # 賦予這個欄位一個新的節點索引
            current_col_idx = len(column_docs)
            column_global_id_to_idx[col_global_id] = current_col_idx
            meta_col_global_ids.append(col_global_id)

            # ('Table', 'has_column', 'Column')
            # (Table 節點索引) -> (Column 節點索引)
            edge_table_to_col_src.append(current_table_idx)
            edge_table_to_col_dst.append(current_col_idx)

            # Column 的簡述
            sample_values = [row[col_idx] for row in instance_rows[:10] if len(row) > col_idx and row[col_idx].strip()]
            col_doc_parts = [
                f"Column: {col_name}",
                f"Belongs to table: {page_title} - {item.get('sheet_name', '')}",
                f"Values: {', '.join(sample_values)}"
            ]
            column_docs.append(" ".join(filter(None, col_doc_parts)))
    
    except Exception as e:
        print(f"處理表格 {item.get('id')} 時發生錯誤: {e}")
        continue

print(f"\n--- 建圖 (共 {len(table_docs)} 表, {len(column_docs)} 欄, {len(page_docs)} 頁) ---")

# 使用 embedder 將所有簡述轉為向量
print("開始 embeddings（table, column, page）...")
data['table'].x = torch.tensor(embedder.encode(table_docs, show_progress_bar=True), dtype=torch.float)
data['column'].x = torch.tensor(embedder.encode(column_docs, show_progress_bar=True), dtype=torch.float)
data['page'].x = torch.tensor(embedder.encode(page_docs, show_progress_bar=True), dtype=torch.float)
print("embeddings 完成。")

# 儲存原始資料
data['table'].id = meta_table_ids
data['column'].global_id = meta_col_global_ids
data['page'].title = meta_page_titles

# 將所有 ID-to-Index 映射集中儲存到頂層字典
data.metadata_maps = {
    'table_id_to_idx': table_id_to_idx,
    'column_global_id_to_idx': column_global_id_to_idx,
    'page_title_to_idx': page_title_to_idx
}

# 儲存結構邊
data['table', 'has_column', 'column'].edge_index = torch.tensor([edge_table_to_col_src, edge_table_to_col_dst], dtype=torch.long)
data['table', 'comes_from', 'page'].edge_index = torch.tensor([edge_table_to_page_src, edge_table_to_page_dst], dtype=torch.long)

# 建立同一 Page 下的 Table 之間的 strong edges (('table','same_page','table'))
print("正在建立同一 page 的 table 連邊 (same_page)...")
same_src = []
same_dst = []
for page_title, t_idxs in page_to_table_idxs.items():
    if len(t_idxs) <= 1:
        continue
    # 兩兩配對，加入雙向邊
    for i in range(len(t_idxs)):
        for j in range(i+1, len(t_idxs)):
            a = t_idxs[i]
            b = t_idxs[j]
            same_src.append(a)
            same_dst.append(b)
            same_src.append(b)  # 加入反向
            same_dst.append(a)

if same_src:
    data['table', 'same_page', 'table'].edge_index = torch.tensor([same_src, same_dst], dtype=torch.long)
    print(f"建立 same_page edges: {len(same_src)} 條（含反向）")
else:
    print("沒有發現同一 page 下有多於 1 張 table，未建立 same_page edges。")

# 建立反向邊
print("正在建立反向邊 (to_homogeneous -> to_heterogeneous)...")
data = data.to_homogeneous().to_heterogeneous()

# 利用相似度來接表格（table <-> table by embedding）
print("計算 'table' <-> 'table' 相似度邊...")
K_table = 5 # 每張表挑 top@5 連結
xt = F.normalize(data['table'].x, p=2, dim=1) # 將向量化的table標準化
sim_matrix = torch.matmul(xt, xt.T) # 計算 N x N 相似度矩陣
sim_matrix.fill_diagonal_(float('-inf')) # 把對角線設為負無窮大以避開自己
top_k_values, top_k_indices = torch.topk(sim_matrix, k=min(K_table, xt.size(0)-1), dim=1) # 找出 topk 個鄰居

sim_src, sim_dst = [], []
for i in range(xt.size(0)):
    for j in top_k_indices[i]:
        sim_src.append(i)
        sim_dst.append(j.item())
        
data['table', 'similar_table', 'table'].edge_index = torch.tensor([sim_src, sim_dst], dtype=torch.long)

print("計算 'column' <-> 'column' 相似度邊...")
K_column = 5 # 每個欄位連接 5 個最像的
xc = F.normalize(data['column'].x, p=2, dim=1)

sim_matrix_c = torch.matmul(xc, xc.T)
sim_matrix_c.fill_diagonal_(float('-inf'))
top_k_values_c, top_k_indices_c = torch.topk(sim_matrix_c, k=min(K_column, xc.size(0)-1), dim=1)

sim_src_c, sim_dst_c = [], []
for i in range(xc.size(0)):
    for j in top_k_indices_c[i]:
        sim_src_c.append(i)
        sim_dst_c.append(j.item())
data['column', 'similar_content', 'column'].edge_index = torch.tensor([sim_src_c, sim_dst_c], dtype=torch.long)

print("\n--- 圖構建完成！ ---")
print("\n最終圖結構 (HeteroData):")
print(data)

# --- 儲存圖 ---
OUTPUT_GRAPH_PATH = "/user_data/TabGNN/data/processed/graph.pt"
print(f"\n將圖儲存至 {OUTPUT_GRAPH_PATH}...")
torch.save(data, OUTPUT_GRAPH_PATH)
print("完成！")
