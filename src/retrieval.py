"""GNN 表格檢索模組"""
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from typing import Tuple, List

from train_model import DiffusionModel, get_embedder

# ========= 可調參數 =========
DEFAULT_MODEL_PATH = "/user_data/TabGNN/checkpoints/model.pt"
DEFAULT_GRAPH_PATH = "/user_data/TabGNN/data/processed/graph_evaluate.pt"
TOP_K = 10
# ===========================


def retrieve(
    query: str,
    top_k: int = TOP_K,
    model_path: str = DEFAULT_MODEL_PATH,
    graph_path: str = DEFAULT_GRAPH_PATH
) -> List[Tuple[int, float]]:
    """執行表格檢索，返回 Top K 的 (表格ID, 分數) 列表"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 載入圖結構
    try:
        data = torch.load(graph_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"錯誤：找不到圖檔案 {graph_path}。")
        return []

    try:
        idx_to_id = {v: k for k, v in data.metadata_maps['table_id_to_idx'].items()}
    except (AttributeError, KeyError):
        print("錯誤：無法從 graph.pt 中載入 ID 映射表。")
        return []

    # 載入模型
    try:
        checkpoint = torch.load(model_path, map_location=device)
        hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2, 'AGGR': 'sum'})
    except FileNotFoundError:
        print(f"錯誤：找不到模型權重 {model_path}。")
        return []

    model = DiffusionModel(
        embed_dim=1024,
        hidden_channels=hps.get('HIDDEN_CHANNELS', 128),
        metadata=data.metadata(),
        dropout=hps.get('DROPOUT', 0.2),
        sage_aggr=hps.get('SAGE_AGGR', 'mean'),
        hetero_aggr=hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 嵌入查詢
    embedder = get_embedder(device='cpu')
    print(f"\n查詢: {query}")
    query_vec = embedder.encode([query], convert_to_numpy=False, convert_to_tensor=True).to(device)

    # 執行檢索
    with torch.no_grad():
        scores = model.score_tables(data.to(device), query_vec)
        k_val = min(top_k, scores.numel())
        top_scores, top_indices = torch.topk(scores, k=k_val, largest=True)

    results = []
    for idx, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()):
        table_id = idx_to_id.get(idx, f"UNKNOWN_{idx}")
        results.append((table_id, score))

    return results


def main():
    """互動式檢索模式"""
    print("互動模式：輸入查詢（輸入 exit/quit 結束）")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        results = retrieve(query, top_k=TOP_K)
        print(f"\n===== 檢索結果 (Top {len(results)}) =====")
        for rank, (table_id, score) in enumerate(results, 1):
            print(f"{rank}. 表格 ID: {table_id} | 分數: {score:.4f}")
        print("========================================\n")


if __name__ == '__main__':
    main()