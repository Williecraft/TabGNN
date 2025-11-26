import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.nn import GraphNorm
from sentence_transformers import SentenceTransformer
import os
import sys
from typing import Tuple, List

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

def retrieve(query: str, top_k: int = 10, model_path: str = "diffusion_mode.pt", graph_path: str = "graph.pt") -> List[Tuple[int, float]]:
    """
    執行表格檢索並返回 Top K 的表格 ID 及其分數。
    
    Args:
        query (str): 查詢問題。
        top_k (int): 返回前 K 個結果。
        model_path (str): 訓練好的模型權重路徑。
        graph_path (str): 訓練好的圖結構路徑。
        
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
        idx_to_id = {v: k for k, v in data.metadata_maps['table_id_to_idx'].items()}
    except (AttributeError, KeyError):
        print("錯誤：無法從 graph.pt 中載入 ID 映射表。請檢查 build_graph.py 是否已修正並重新執行。")
        return []
    
    # 2. 載入模型和超參數
    try:
        checkpoint = torch.load(model_path, map_location=device)
        hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2}) # 向下相容
        hidden_channels = hps.get('HIDDEN_CHANNELS', 128)
        dropout = hps.get('DROPOUT', 0.2)

    except FileNotFoundError:
        print(f"錯誤：找不到模型權重 {model_path}。請確認訓練已完成且檔案存在。")
        return []

    # 3. 初始化模型
    model = DiffusionModel(
        embed_dim=embedder, 
        hidden_channels=hidden_channels, 
        metadata=data.metadata(),
        dropout=dropout
    ).to(device)
    
    # 載入模型權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 設置為評估模式
    
    # 3. 嵌入查詢
    embedder = get_embedder(device='cpu') # 嵌入模型通常在 CPU 上執行更快或更方便
    
    print(f"\n查詢: {query}")
    query_vec_np = embedder.encode([query], convert_to_numpy=False, convert_to_tensor=True)
    query_vec = query_vec_np.to(device)

    # 4. 執行檢索
    with torch.no_grad():
        # 將圖數據移至檢索設備
        data_on_device = data.to(device)
        
        # 獲取所有表格的分數 (Scores: N)
        scores = model.score_tables(data_on_device, query_vec)
        
        # 找出 Top K
        k_val = min(top_k, scores.numel())
        top_scores, top_indices = torch.topk(scores, k=k_val, largest=True)

    # 5. 格式化結果
    results = []
    for idx, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()):
        table_id = idx_to_id.get(idx, f"UNKNOWN_ID_IDX_{idx}")
        results.append((table_id, score))
        
    return results

def main():
    
    DEFAULT_MODEL_PATH = "diffusion_model.pt" 
    DEFAULT_GRAPH_PATH = "graph.pt"
    TOP_K = 10

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
        retrieved_tables = retrieve(
            query=query, 
            top_k=TOP_K, 
            model_path=DEFAULT_MODEL_PATH,
            graph_path=DEFAULT_GRAPH_PATH
        )

        # 輸出結果
        print(f"\n===== 檢索結果 (Top {len(retrieved_tables)}) =====")
        for rank, (table_id, score) in enumerate(retrieved_tables, 1):
            print(f"{rank}. 表格 ID: {table_id} | 相似度分數: {score:.4f}")
        print("========================================\n")

if __name__ == '__main__':
    main()