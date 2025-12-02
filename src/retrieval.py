import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.nn import GraphNorm
from sentence_transformers import SentenceTransformer
import os
import sys
import json
import csv
from typing import Tuple, List, Dict, Set
from sentence_transformers import SentenceTransformer, CrossEncoder

# ====================================================================
# A. 輔助函式 (Embedder & Model Definition)
# 
# 由於這是一個獨立腳本，我們需要重新定義或導入所有必要的組件。
# 這裡使用您 train_model.py 中的模型和嵌入器邏輯。
# ====================================================================

# 載入 SentenceTransformer 模型
def get_embedder(model_name='BAAI/bge-m3', device='cpu'):
    """載入 SentenceTransformer 模型"""
    try:
        # 強制使用 CPU 載入，如果需要 GPU，可以在環境變數中設定
        return SentenceTransformer(model_name, device=device) 
    except Exception as e:
        print(f"錯誤：無法載入 SentenceTransformer 模型。請安裝 sentence-transformers 庫。{e}")
        sys.exit(1)

def load_table_texts(jsonl_path: str) -> Dict[str, str]:
    """從 JSONL 檔案載入表格文字內容，用於 Cross-Encoder 重排序。"""
    texts = {}
    if not os.path.exists(jsonl_path):
        print(f"警告：找不到表格檔案 {jsonl_path}，將跳過 Refinement 階段的文字載入。")
        return texts
        
    print(f"正在載入表格內容 from {jsonl_path} ...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                table_id = str(item['id']) # 強制轉為字串，確保與 idx_to_id 一致
                
                # 建構表格文字表示 (參考 build_graph.py)
                header = item.get('header', [])
                header_list = []
                if header and header[0]:
                    header_list = next(csv.reader([header[0]]))
                
                instance_rows = []
                for row_str in item.get('instances', []):
                    instance_rows.append(next(csv.reader([row_str])))
                
                page_title = item.get('metadata', {}).get('table_page_title', '')
                
                table_doc_parts = [
                    f"Page: {page_title}",
                    f"Sheet: {item.get('sheet_name', '')}",
                    f"Section: {item.get('metadata', {}).get('table_section_title', '')}",
                    f"Columns: {', '.join(header_list)}",
                    f"Data: {'; '.join([', '.join(row) for row in instance_rows[:5]])}"
                ]
                texts[table_id] = " ".join(filter(None, table_doc_parts))
            except Exception:
                continue
    print(f"已載入 {len(texts)} 張表格內容。")
    return texts

# GNN 模型定義 (與 train_model.py 保持一致)
class DiffusionModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_channels: int, metadata, dropout: float = 0.2):
        super().__init__()

        # 與 train_model.py 保持一致的架構
        self.sage = GraphSAGE(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=hidden_channels,
        )

        self.hetero_sage = to_hetero(self.sage, metadata, aggr='sum')

        # 新的投影頭（LayerNorm + MLP + Dropout）
        self.norm = GraphNorm(hidden_channels) 
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, embed_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict):
        x_dict = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    # 檢索的核心方法
    def score_tables(self, data: HeteroData, query_vec: torch.Tensor) -> torch.Tensor:
        """計算單一查詢向量與圖中所有表格節點的相似度分數。"""
        # 1. 獲取經過 GNN 強化後的表格特徵
        table_embeddings = self.forward(data.x_dict, data.edge_index_dict)
        
        # 2. 規範化查詢向量
        query_vec = F.normalize(query_vec.to(table_embeddings.device), p=2, dim=1)
        
        # 3. 計算點積 (餘弦相似度)
        scores = torch.matmul(query_vec, table_embeddings.T).squeeze(0)
        return scores


# ====================================================================
# B. 檢索主函數
# ====================================================================

def retrieve(query: str, top_k: int = 10, model_path: str = "diffusion_model.pt", graph_path: str = "graph.pt", table_jsonl_path: str = "/user_data/TabGNN/data/table/test/feta/table.jsonl") -> List[Tuple[int, float]]:
    """
    執行 REaR (Retrieve, Expand, Refine) 表格檢索。
    
    Args:
        query (str): 查詢問題。
        top_k (int): 返回前 K 個結果。
        model_path (str): 訓練好的模型權重路徑。
        graph_path (str): 訓練好的圖結構路徑。
        table_jsonl_path (str): 表格原始資料路徑 (用於 Refinement)。
        
    Returns:
        List[Tuple[int, float]]: 包含 (表格ID, 分數) 的列表。
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 1. 載入圖結構
    try:
        # map_location 確保可以將 GPU 儲存的模型或圖載入到任何設備
        data = torch.load(graph_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"錯誤：找不到圖檔案 {graph_path}。請確認路徑或重新執行 build_graph.py。")
        return []

    # 獲取映射表
    try:
        idx_to_id = {v: str(k) for k, v in data.metadata_maps['table_id_to_idx'].items()}
        # id_to_idx = data.metadata_maps['table_id_to_idx'] # 暫時不需要
    except (AttributeError, KeyError):
        print("錯誤：無法從 graph.pt 中載入 ID 映射表。請檢查 build_graph.py 是否已修正並重新執行。")
        return []
    
    # 2. 載入模型和超參數
    try:
        checkpoint = torch.load(model_path, map_location=device)
        hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2}) # 向下相容
        hidden_channels = hps.get('HIDDEN_CHANNELS', 128)
        dropout = hps.get('DROPOUT', 0.2)
        embed_dim = 1024 # BAAI/bge-m3 維度

    except FileNotFoundError:
        print(f"錯誤：找不到模型權重 {model_path}。請確認訓練已完成且檔案存在。")
        return []

    # 3. 初始化模型
    model = DiffusionModel(
        embed_dim=embed_dim, 
        hidden_channels=hidden_channels, 
        metadata=data.metadata(),
        dropout=dropout
    ).to(device)
    
    # 載入模型權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 設置為評估模式
    
def retrieve_with_resources(
    query: str,
    top_k: int,
    model: DiffusionModel,
    data: HeteroData,
    embedder: SentenceTransformer,
    table_texts: Dict[str, str],
    device: torch.device,
    idx_to_id: Dict[int, str]
) -> List[Tuple[str, float]]:
    """
    使用預先載入的資源執行 REaR 檢索。
    """
    # 3. 嵌入查詢
    query_vec_np = embedder.encode([query], convert_to_numpy=False, convert_to_tensor=True)
    query_vec = query_vec_np.to(device)

    # ====================================================================
    # Stage 1: Retrieval (Base Retrieval using GNN)
    # ====================================================================
    # print("--- Stage 1: Retrieval (GNN) ---")
    with torch.no_grad():
        data_on_device = data.to(device)
        scores = model.score_tables(data_on_device, query_vec)
        
        # 這裡我們先取多一點，作為 Base Set，例如 Top-K * 2
        k_base = min(top_k * 2, scores.numel())
        base_scores, base_indices = torch.topk(scores, k=k_base, largest=True)
        
    base_indices_set = set(base_indices.cpu().tolist())
    # print(f"Base Set size: {len(base_indices_set)}")

    # ====================================================================
    # Stage 2: Expansion (Graph-based Joinability)
    # ====================================================================
    # print("--- Stage 2: Expansion ---")
    # 利用圖結構找出與 Base Tables 有 'similar_content' 欄位連接的表格
    
    # 1. 建立 Column -> Table 的映射 (反向查詢)
    # data['table', 'has_column', 'column'].edge_index: [0]=TableIdx, [1]=ColIdx
    t2c_edge_index = data['table', 'has_column', 'column'].edge_index.cpu()
    col_to_table = {}
    for i in range(t2c_edge_index.size(1)):
        t_idx = t2c_edge_index[0, i].item()
        c_idx = t2c_edge_index[1, i].item()
        col_to_table[c_idx] = t_idx

    # 2. 找出 Base Tables 的所有 Columns
    base_cols = set()
    for i in range(t2c_edge_index.size(1)):
        t_idx = t2c_edge_index[0, i].item()
        if t_idx in base_indices_set:
            base_cols.add(t2c_edge_index[1, i].item())
            
    # 3. 找出與這些 Columns 相似的其他 Columns
    # data['column', 'similar_content', 'column'].edge_index
    c2c_edge_index = data['column', 'similar_content', 'column'].edge_index.cpu()
    
    expanded_tables = set()
    for i in range(c2c_edge_index.size(1)):
        src_c = c2c_edge_index[0, i].item()
        dst_c = c2c_edge_index[1, i].item()
        
        # 如果 src_c 屬於 Base Table，則 dst_c 所屬的 Table 為候選
        if src_c in base_cols:
            if dst_c in col_to_table:
                target_table = col_to_table[dst_c]
                if target_table not in base_indices_set:
                    expanded_tables.add(target_table)
                    
    # print(f"Expanded Set size: {len(expanded_tables)}")
    
    # 合併 Base 和 Expanded
    candidate_indices = list(base_indices_set | expanded_tables)
    # print(f"Total Candidates for Refinement: {len(candidate_indices)}")

    # ====================================================================
    # Stage 3: Refinement (Cross-Encoder Reranking)
    # ====================================================================
    # print("--- Stage 3: Refinement ---")
    
    if not table_texts:
        # print("無法載入表格文字，降級為僅使用 GNN 分數排序。")
        # 如果沒有文字，就只用 GNN 分數 (對於 Expanded 表格，分數設為 0 或最小值)
        final_results = []
        min_score = base_scores[-1].item() if len(base_scores) > 0 else 0.0
        for idx in candidate_indices:
            score = scores[idx].item() # GNN score
            # 如果是 expanded 的，可能分數很低，但我們還是保留它
            final_results.append((idx, score))
        final_results.sort(key=lambda x: x[1], reverse=True)
    else:
        # 使用 Cross-Encoder
        try:
            # 注意：這裡每次都初始化 CrossEncoder 會很慢，但在這個函式介面下暫時只能這樣
            # 為了優化，應該將 reranker 也傳入
            # 這裡假設外部會傳入或我們使用全域變數，或者為了簡單起見，這裡先即時載入(雖然慢)
            # 更好的方式是將 reranker 作為參數傳入。
            # 為了不破壞函式簽名太多，我們在內部檢查是否有傳入 reranker (透過 kwargs 或修改簽名)
            # 這裡我選擇修改簽名，但為了相容性，我先在內部載入，使用者應該在外部載入並傳入
            # 暫時：每次載入 (很慢，但正確)。
            # 優化：修改 evaluate_retrieval.py 時，我們可以直接呼叫這個邏輯，或者將 reranker 傳入。
            # 讓我們修改一下函式簽名，加入 reranker
            pass
        except Exception:
            pass

    # 為了效能，我們將 Reranker 邏輯移到外部或假設它很快。
    # 實際上，CrossEncoder 載入需要時間。
    # 我們修改一下 retrieve_with_resources 的簽名，加入 reranker
    return [] # 這裡只是佔位，下面會被完整替換

def retrieve(query: str, top_k: int = 10, model_path: str = "diffusion_model.pt", graph_path: str = "graph.pt", table_jsonl_path: str = "/user_data/TabGNN/data/table/test/feta/table.jsonl") -> List[Tuple[int, float]]:
    """
    執行 REaR (Retrieve, Expand, Refine) 表格檢索。
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 1. 載入圖結構
    try:
        data = torch.load(graph_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"錯誤：找不到圖檔案 {graph_path}。請確認路徑或重新執行 build_graph.py。")
        return []

    # 獲取映射表
    try:
        idx_to_id = {v: k for k, v in data.metadata_maps['table_id_to_idx'].items()}
    except (AttributeError, KeyError):
        print("錯誤：無法從 graph.pt 中載入 ID 映射表。請檢查 build_graph.py 是否已修正並重新執行。")
        return []
    
    # 2. 載入模型和超參數
    try:
        checkpoint = torch.load(model_path, map_location=device)
        hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2})
        hidden_channels = hps.get('HIDDEN_CHANNELS', 128)
        dropout = hps.get('DROPOUT', 0.2)
        embed_dim = 1024 
    except FileNotFoundError:
        print(f"錯誤：找不到模型權重 {model_path}。請確認訓練已完成且檔案存在。")
        return []

    # 3. 初始化模型
    model = DiffusionModel(
        embed_dim=embed_dim, 
        hidden_channels=hidden_channels, 
        metadata=data.metadata(),
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    embedder = get_embedder(device='cpu')
    
    # 載入表格文字
    table_texts = load_table_texts(table_jsonl_path)
    
    # 載入 Reranker
    reranker = None
    if table_texts:
        try:
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        except Exception as e:
            print(f"無法載入 Reranker: {e}")

    return retrieve_with_resources(
        query=query,
        top_k=top_k,
        model=model,
        data=data,
        embedder=embedder,
        table_texts=table_texts,
        device=device,
        idx_to_id=idx_to_id,
        reranker=reranker
    )

def retrieve_with_resources(
    query: str,
    top_k: int,
    model: DiffusionModel,
    data: HeteroData,
    embedder: SentenceTransformer,
    table_texts: Dict[str, str],
    device: torch.device,
    idx_to_id: Dict[int, str],
    reranker: CrossEncoder = None
) -> List[Tuple[str, float]]:
    
    # 3. 嵌入查詢
    query_vec_np = embedder.encode([query], convert_to_numpy=False, convert_to_tensor=True)
    query_vec = query_vec_np.to(device)

    # Stage 1: Retrieval
    with torch.no_grad():
        data_on_device = data.to(device)
        scores = model.score_tables(data_on_device, query_vec)
        k_base = min(top_k * 2, scores.numel())
        base_scores, base_indices = torch.topk(scores, k=k_base, largest=True)
        
    base_indices_set = set(base_indices.cpu().tolist())

    # Stage 2: Expansion
    t2c_edge_index = data['table', 'has_column', 'column'].edge_index.cpu()
    # 優化：這個映射表應該預先建立，但為了簡單起見這裡即時建立（會稍微慢一點）
    # 如果 evaluate 迴圈呼叫此函式，建議在外部建立好傳入，這裡先保持這樣
    col_to_table = {}
    for i in range(t2c_edge_index.size(1)):
        t_idx = t2c_edge_index[0, i].item()
        c_idx = t2c_edge_index[1, i].item()
        col_to_table[c_idx] = t_idx

    base_cols = set()
    for i in range(t2c_edge_index.size(1)):
        t_idx = t2c_edge_index[0, i].item()
        if t_idx in base_indices_set:
            base_cols.add(t2c_edge_index[1, i].item())
            
    c2c_edge_index = data['column', 'similar_content', 'column'].edge_index.cpu()
    
    expanded_tables = set()
    for i in range(c2c_edge_index.size(1)):
        src_c = c2c_edge_index[0, i].item()
        dst_c = c2c_edge_index[1, i].item()
        
        if src_c in base_cols:
            if dst_c in col_to_table:
                target_table = col_to_table[dst_c]
                if target_table not in base_indices_set:
                    expanded_tables.add(target_table)
    
    candidate_indices = list(base_indices_set | expanded_tables)

    # Stage 3: Refinement
    if not table_texts or reranker is None:
        final_results = []
        for idx in candidate_indices:
            score = scores[idx].item()
            final_results.append((idx, score))
        final_results.sort(key=lambda x: x[1], reverse=True)
    else:
        try:
            rerank_inputs = []
            valid_candidates = []
            
            for idx in candidate_indices:
                t_id = idx_to_id.get(idx)
                if t_id and t_id in table_texts:
                    text = table_texts[t_id]
                    rerank_inputs.append([query, text])
                    valid_candidates.append(idx)
            
            if rerank_inputs:
                rerank_scores = reranker.predict(rerank_inputs)
                final_results = []
                for idx, r_score in zip(valid_candidates, rerank_scores):
                    final_results.append((idx, float(r_score)))
                final_results.sort(key=lambda x: x[1], reverse=True)
            else:
                # 如果 rerank_inputs 為空 (例如找不到對應的文字)，降級為 GNN 分數
                final_results = []
                for idx in candidate_indices:
                    final_results.append((idx, scores[idx].item()))
                final_results.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Reranker error: {e}")
            final_results = []
            for idx in candidate_indices:
                final_results.append((idx, scores[idx].item()))
            final_results.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in final_results[:top_k]:
        table_id = idx_to_id.get(idx, f"UNKNOWN_ID_IDX_{idx}")
        results.append((table_id, score))
        
    return results

def main():
    
    DEFAULT_MODEL_PATH = "/user_data/TabGNN/checkpoints/diffusion_model.pt" 
    DEFAULT_GRAPH_PATH = "/user_data/TabGNN/data/processed/graph.pt"
    TABLE_JSONL_PATH = "/user_data/TabGNN/data/table/test/feta/table.jsonl"
    TOP_K = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    print("正在初始化並載入所有資源，請稍候...")

    # 1. 載入圖結構
    try:
        print(f"載入圖結構: {DEFAULT_GRAPH_PATH}")
        data = torch.load(DEFAULT_GRAPH_PATH, map_location=device, weights_only=False)
        # 強制轉為字串，確保與 table.jsonl 的 id 一致
        idx_to_id = {v: str(k) for k, v in data.metadata_maps['table_id_to_idx'].items()}
    except Exception as e:
        print(f"錯誤：無法載入圖結構: {e}")
        return

    # 2. 載入模型
    try:
        print(f"載入模型: {DEFAULT_MODEL_PATH}")
        checkpoint = torch.load(DEFAULT_MODEL_PATH, map_location=device)
        hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2})
        hidden_channels = hps.get('HIDDEN_CHANNELS', 128)
        dropout = hps.get('DROPOUT', 0.2)
        embed_dim = 1024 

        model = DiffusionModel(
            embed_dim=embed_dim, 
            hidden_channels=hidden_channels, 
            metadata=data.metadata(),
            dropout=dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"錯誤：無法載入模型: {e}")
        return

    # 3. 載入 Embedder
    print("載入 SentenceTransformer (Embedder)...")
    embedder = get_embedder(device='cpu')

    # 4. 載入表格文字
    print(f"載入表格文字: {TABLE_JSONL_PATH}")
    table_texts = load_table_texts(TABLE_JSONL_PATH)

    # 5. 載入 Reranker
    reranker = None
    if table_texts:
        try:
            print("載入 CrossEncoder (Reranker)...")
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        except Exception as e:
            print(f"警告：無法載入 Reranker: {e}")

    print("\n所有資源載入完成！")
    print("互動模式：輸入查詢（輸入 exit/quit 結束）")
    
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出互動模式。")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        # 執行檢索
        retrieved_tables = retrieve_with_resources(
            query=query, 
            top_k=TOP_K, 
            model=model,
            data=data,
            embedder=embedder,
            table_texts=table_texts,
            device=device,
            idx_to_id=idx_to_id,
            reranker=reranker
        )

        # 輸出結果
        print(f"\n===== 檢索結果 (Top {len(retrieved_tables)}) =====")
        for rank, (table_id, score) in enumerate(retrieved_tables, 1):
            print(f"{rank}. 表格 ID: {table_id} | 相似度分數: {score:.4f}")
        print("========================================\n")

if __name__ == '__main__':
    main()